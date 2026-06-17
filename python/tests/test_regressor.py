import numpy as np
from xuplift import Regressor


def test_regressor_fit_predict():
    np.random.seed(42)
    n_samples = 100
    n_features = 2
    x = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (x[:, 0] * 2 + x[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1).astype(
        np.float32
    )

    model = Regressor(max_bases=64, penalty=0.1)
    model.fit(x, y, [False, False])

    preds = model.predict(x)
    assert preds.shape == (n_samples,)

    correlation = np.corrcoef(preds, y)[0, 1]
    assert correlation > 0.8


def test_regressor_explain():
    np.random.seed(42)
    n_samples = 50
    n_features = 3
    x = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (x[:, 0] * 5).astype(np.float32)

    model = Regressor(max_bases=64, penalty=0.1)
    model.fit(x, y, [False, False, False])

    explanation = model.explain(x)
    assert explanation.shape == (n_samples, n_features)
    assert np.any(explanation != 0)
