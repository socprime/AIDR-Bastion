import joblib

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
        status depending on the success of model loading.
        """
        self.model_classifier = self._load_model()
        if self.model_classifier:
            self.enabled = True
            bastion_logger.info(f"[{self}] loaded successfully. Model path: {settings.ML_MODEL_PATH}")
        else:
            bastion_logger.warning(f"[{self}] failed to load model. Model path: {settings.ML_MODEL_PATH}")

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
        except Exception as err:
            bastion_logger.error(f"Error loading model from {settings.ML_MODEL_PATH}, error={err}")

    def validate_prompt(self, prompt: str):
        """
        Validates prompt using ML model.

        Converts text prompt to vector representation and passes it
        to ML model for classification to detect malicious content.

        Args:
            prompt (str): Text prompt for analysis

        Returns:
            Model classification result or None on embedding creation error
        """
        try:
            if embedding := text_embedding(prompt):
                predict = self.model_classifier.predict(embedding)
                return predict
        except Exception as err:
            bastion_logger.warning(f"Error validating prompt, error={str(err)}")

    async def run(self, prompt: str) -> PipelineResult:
        """
        Performs prompt analysis for malicious content.

        Analyzes input prompt using ML model and creates analysis result
        with information about triggered rules. If model detects
        malicious content, adds blocking rule to the result.

        Args:
            prompt (str): Text prompt for analysis

        Returns:
            PipelineResult: Analysis result with list of triggered rules
        """
        trigger_rules = []
        bastion_logger.info(f"Analyzing for {self._identifier}")
        status = ActionStatus.ALLOW
        if self.validate_prompt(prompt):
            msg = "ML Pipeline detected malicious prompt"
            status = ActionStatus.BLOCK
            trigger_rules.append(
                TriggeredRuleData(id=self._identifier, name=str(self), details=msg, action=RuleAction.BLOCK)
            )
            bastion_logger.info(f"Analyzing for {self._identifier}, status: {status}, details: {msg}")
        bastion_logger.info(f"Analyzing done for {self._identifier}")
        return PipelineResult(name=str(self), triggered_rules=trigger_rules, status=status)
