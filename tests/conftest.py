"""Shared fixtures for the AIDR-Bastion test suite.

Importing ``app.pipelines.ml_pipeline.pipeline`` normally drags in the whole
``app.pipelines`` package, whose ``__init__`` instantiates every pipeline at
import time (OpenSearch/Qdrant clients, semgrep, LLM SDKs), and ``app.utils``,
which loads a ~550 MB SentenceTransformer and may reach out to the network.
The suite must run offline and in seconds, so lightweight stand-ins for both
are installed in :data:`sys.modules` before the pipeline module is imported.
"""

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

EMBEDDING_DIM = 768


def _install_import_stubs() -> None:
    """
    Registers stand-ins for the import-time heavy parts of the ``app`` package.

    ``app.utils`` is replaced by a module whose ``text_embedding`` refuses to run
    unpatched, so a test that forgets to patch it fails loudly instead of quietly
    downloading an embeddings model. ``app.pipelines`` is replaced by an empty
    package object that keeps the real ``__path__``, so its submodules still
    import normally while its eager pipeline instantiation is skipped.
    """
    utils_stub = types.ModuleType("app.utils")

    def text_embedding(prompt: str) -> list[float]:
        """
        Placeholder for the real embedding function.

        Args:
            prompt (str): Text prompt that would be embedded.

        Raises:
            RuntimeError: Always, to flag a test that did not patch the embedding.
        """
        raise RuntimeError("text_embedding must be patched in tests, see the patched_embedding fixture")

    utils_stub.text_embedding = text_embedding
    sys.modules.setdefault("app.utils", utils_stub)

    pipelines_stub = types.ModuleType("app.pipelines")
    pipelines_stub.__path__ = [str(REPO_ROOT / "app" / "pipelines")]
    sys.modules.setdefault("app.pipelines", pipelines_stub)


_install_import_stubs()

from app.pipelines.ml_pipeline import pipeline as ml_pipeline  # noqa: E402  (needs the stubs above)


class FakeClassifier:
    """
    Minimal stand-in for a fitted scikit-learn estimator.

    Validates the shape of every input the way a real estimator does: anything
    that is not a 2-D ``(n_samples, n_features)`` array is rejected with a
    ``ValueError``. Without that check a pipeline handing a flat vector to
    ``predict`` would look healthy in tests while failing in production.

    Attributes:
        prediction (int): Label returned for every sample (1 = malicious).
        calls (list[np.ndarray]): Every array accepted by predict/predict_proba.
    """

    def __init__(self, prediction: int = 1) -> None:
        """
        Initializes the classifier with a fixed label.

        Args:
            prediction (int): Label returned for every sample.
        """
        self.prediction = prediction
        self.calls: list[np.ndarray] = []

    def _as_2d(self, X: Any) -> np.ndarray:
        """
        Coerces the input to an array and records it, rejecting non-2-D input.

        Args:
            X (Any): Feature matrix passed by the caller.

        Returns:
            np.ndarray: The validated feature matrix.

        Raises:
            ValueError: If the input is not a 2-D array.
        """
        features = np.asarray(X)
        if features.ndim != 2:
            raise ValueError(f"Expected 2D array, got {features.ndim}D array instead")
        self.calls.append(features)
        return features

    def predict(self, X: Any) -> np.ndarray:
        """
        Returns the configured label for every sample.

        Args:
            X (Any): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Array of shape (n_samples,) filled with the configured label.
        """
        features = self._as_2d(X)
        return np.full(features.shape[0], self.prediction)

    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Returns degenerate class probabilities matching the configured label.

        Args:
            X (Any): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Array of shape (n_samples, 2) with the benign and malicious scores.
        """
        features = self._as_2d(X)
        malicious = float(self.prediction)
        return np.tile([1.0 - malicious, malicious], (features.shape[0], 1))


@pytest.fixture
def fake_classifier() -> FakeClassifier:
    """
    Provides a classifier that labels every prompt as malicious.

    Returns:
        FakeClassifier: Classifier recording the arrays it receives.
    """
    return FakeClassifier(prediction=1)


@pytest.fixture
def patched_embedding(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """
    Replaces ``text_embedding`` with a deterministic vector.

    The real function returns a flat ``list[float]`` of length 768
    (``model.encode(...).tolist()``); the stand-in reproduces exactly that shape
    so tests exercise the same conversion path as production.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture used to patch the module attribute.

    Returns:
        list[float]: The vector the pipeline will receive.
    """
    embedding = [0.01 * (index % 100) for index in range(EMBEDDING_DIM)]
    monkeypatch.setattr(ml_pipeline, "text_embedding", lambda prompt: list(embedding))
    return embedding


@pytest.fixture
def configured_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_classifier: FakeClassifier
) -> FakeClassifier:
    """
    Configures ``ML_MODEL_PATH`` and makes ``joblib.load`` return the fake classifier.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture used to patch settings and joblib.
        tmp_path (Path): Temporary directory providing a plausible model path.
        fake_classifier (FakeClassifier): Classifier the loader hands back.

    Returns:
        FakeClassifier: The classifier a correctly wired pipeline should end up with.
    """
    model_path = tmp_path / "classifier.joblib"
    model_path.touch()
    monkeypatch.setattr(ml_pipeline.settings, "ML_MODEL_PATH", str(model_path))
    monkeypatch.setattr(ml_pipeline.joblib, "load", lambda path: fake_classifier)
    return fake_classifier


@pytest.fixture
def unconfigured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Clears ``ML_MODEL_PATH`` so the pipeline has nothing to load.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture used to patch the setting.
    """
    monkeypatch.setattr(ml_pipeline.settings, "ML_MODEL_PATH", None)
