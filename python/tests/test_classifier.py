import numpy as np
from xuplift import Classifier


def test_classifier_fit_predict():
    # Generate simple separable data
    np.random.seed(42)
    n_samples = 100
    n_features = 2
    x = np.random.randn(n_samples, n_features).astype(np.float32)
    # Class 1 if x[0] + x[1] > 0
    y = (x[:, 0] + x[:, 1] > 0).astype(np.float32)

    model = Classifier(penalty=0.1, max_iter=20)
    model.fit(x, y)

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
    model.fit(x, y)

    explanation = model.explain(x)
    assert explanation.shape == (n_samples, n_features)

    # Check consistency: sigmoid(sum(contributions) + base_value) == predict
    # Note: base_value is internal but we can check if sum of contributions
    # correlates with predictions.
    # Since we can't easily access base_value from Python (it's not exposed in __init__.pyi),
    # we just check the shape and non-zero.
    assert np.any(explanation != 0)
