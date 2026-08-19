import joblib
import numpy as np

from app.core.enums import ActionStatus, PipelineNames, RuleAction
from app.core.pipeline import BasePipeline
from app.models.pipeline import PipelineResult, TriggeredRuleData
from app.modules.logger import bastion_logger
from app.utils import text_embedding
from settings import get_settings

settings = get_settings()


class MLPipeline(BasePipeline):
    """
    Machine learning-based pipeline for detecting malicious prompts.

    This pipeline uses a pre-trained machine learning model to analyze prompts
    and detect potentially malicious content. The model works with vector
    representations of text (embeddings) for classification.

    A pipeline without a configured model stays disabled and allows every
    prompt: that is the only legitimate allow-by-default case. A pipeline that
    is enabled but fails while classifying fails closed to NOTIFY and never
    silently allows, so a broken detector stays visible to the operator instead
    of looking like a clean verdict.

    Attributes:
        _identifier (PipelineNames): Pipeline identifier (ml)
        model_classifier: Loaded machine learning model
        enabled (bool): Whether pipeline is active (depends on successful model loading)
    """

    _identifier = PipelineNames.ml
    description = "Machine learning-based pipeline for detecting malicious prompts."

    def __init__(self):
        """
        Initializes ML pipeline and loads the classification model.

        Loads a pre-trained model from file and sets the pipeline's active
        status depending on the success of model loading. A missing
        ML_MODEL_PATH is a deliberate opt-out; a configured path that fails to
        load is an error, not a warning.
        """
        self.model_classifier = self._load_model()
        if self.model_classifier:
            self.enabled = True
            bastion_logger.info(f"[{self}] loaded successfully. Model path: {settings.ML_MODEL_PATH}")
        elif not settings.ML_MODEL_PATH:
            bastion_logger.info(f"[{self}] disabled, no ML_MODEL_PATH configured")
        else:
            bastion_logger.error(f"[{self}] failed to load model. Model path: {settings.ML_MODEL_PATH}")

    def __str__(self) -> str:
        return "ML Pipeline"

    def _load_model(self):
        """
        Loads machine learning model from file.

        Uses joblib to load the saved model from the path specified in
        settings. Returns None in case of error.

        Returns:
            Classification model or None on loading error
        """
        if not settings.ML_MODEL_PATH:
            return None
        try:
            return joblib.load(settings.ML_MODEL_PATH)
        except Exception:
            bastion_logger.exception(f"Error loading model from {settings.ML_MODEL_PATH}")

    def validate_prompt(self, prompt: str) -> bool:
        """
        Classifies the prompt as malicious or benign.

        Converts text prompt to vector representation and passes it
        to ML model for classification to detect malicious content.

        The classifier expects a 2-D (1, n_features) array while text_embedding
        returns a flat list, so the embedding is reshaped before prediction.
        float32 is deliberate: scikit-learn casts to it anyway, so doing it here
        avoids an extra float64 copy on the hot path. Embeddings arrive already
        normalized (normalize_embeddings=True), which makes cosine distance
        between them a plain dot product.

        Args:
            prompt (str): Text prompt for analysis

        Returns:
            bool: True if the model labels the prompt as malicious

        Raises:
            ValueError: If the embedding is empty, so the prompt cannot be classified
            Exception: Any failure raised by the embedding backend or the classifier;
                handling belongs to run(), which must not turn it into an allow
        """
        embedding = text_embedding(prompt)
        if not embedding:
            raise ValueError(f"Empty embedding, cannot classify prompt of length {len(prompt)}")
        features = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        prediction = self.model_classifier.predict(features)
        return bool(prediction[0])

    async def run(self, prompt: str) -> PipelineResult:
        """
        Performs prompt analysis for malicious content.

        Analyzes input prompt using ML model and creates analysis result
        with information about triggered rules. If model detects
        malicious content, adds blocking rule to the result.

        A disabled pipeline allows the prompt, which is the only legitimate
        allow-by-default case. A classification failure fails closed to NOTIFY
        with a triggered rule describing the failure, and never silently allows.

        Args:
            prompt (str): Text prompt for analysis

        Returns:
            PipelineResult: Analysis result with list of triggered rules
        """
        bastion_logger.info(f"Analyzing for {self._identifier}")
        if not self.enabled:
            bastion_logger.debug(f"Analyzing skipped for {self._identifier}, pipeline is disabled")
            return PipelineResult(name=str(self), triggered_rules=[], status=ActionStatus.ALLOW)

        trigger_rules = []
        status = ActionStatus.ALLOW
        try:
            is_malicious = self.validate_prompt(prompt)
        except Exception:
            msg = "ML Pipeline failed to classify the prompt"
            bastion_logger.exception(f"{msg} for {self._identifier}")
            # NOTIFY rather than BLOCK: our own failure must stay visible without becoming a self-inflicted DoS.
            return PipelineResult(
                name=str(self),
                triggered_rules=[
                    TriggeredRuleData(id=self._identifier, name=str(self), details=msg, action=RuleAction.NOTIFY)
                ],
                status=ActionStatus.NOTIFY,
            )

        if is_malicious:
            msg = "ML Pipeline detected malicious prompt"
            status = ActionStatus.BLOCK
            trigger_rules.append(
                TriggeredRuleData(id=self._identifier, name=str(self), details=msg, action=RuleAction.BLOCK)
            )
            bastion_logger.info(f"Analyzing for {self._identifier}, status: {status}, details: {msg}")
        bastion_logger.info(f"Analyzing done for {self._identifier}")
        return PipelineResult(name=str(self), triggered_rules=trigger_rules, status=status)
