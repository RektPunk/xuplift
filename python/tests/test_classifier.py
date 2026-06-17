import numpy as np
from xuplift import Classifier


def test_classifier_fit_predict():
    np.random.seed(42)
    n_samples = 100
    n_features = 2
    x = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (x[:, 0] + x[:, 1] > 0).astype(np.float32)  # Class 1 if x[0] + x[1] > 0

    model = Classifier(penalty=0.1, max_iter=20)
    model.fit(x, y, [False, False])

    probs = model.predict(x)
    preds = (probs > 0.5).astype(np.float32)

    accuracy = np.mean(preds == y)
    assert accuracy > 0.8


def test_classifier_explain():
    np.random.seed(42)
    n_samples = 50
    n_features = 3
    x = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.float32)

    model = Classifier(penalty=0.1, max_iter=10)
    model.fit(x, y, [False, False, False])

    explanation = model.explain(x)
    assert explanation.shape == (n_samples, n_features)
    assert np.any(explanation != 0)
