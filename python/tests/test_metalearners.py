import numpy as np
import pytest
from xuplift import RLearner, SLearner, TLearner, XLearner


@pytest.fixture
def uplift_data():
    np.random.seed(42)
    n_samples = 200
    n_features = 3
    x = np.random.randn(n_samples, n_features).astype(np.float32)
    t = np.random.randint(0, 2, n_samples).astype(np.float32)
    # Uplift depends on x[0]
    # y = base_effect + t * uplift + noise
    uplift = x[:, 0] * 2
    y = (x[:, 1] + t * uplift + np.random.randn(n_samples) * 0.1).astype(np.float32)
    return x, t, y


def test_slearner(uplift_data):
    x, t, y = uplift_data
    model = SLearner(x, t, y, mu_penalty=0.1)

    ite = model.predict_uplift(x)
    assert ite.shape == (x.shape[0],)

    explanation = model.explain_uplift(x)
    # SLearner explain_uplift returns (n_samples, n_features + 1)
    assert explanation.shape == (x.shape[0], x.shape[1] + 1)


def test_tlearner(uplift_data):
    x, t, y = uplift_data
    model = TLearner(x, t, y, mu_penalty=0.1)

    ite = model.predict_uplift(x)
    assert ite.shape == (x.shape[0],)

    explanation = model.explain_uplift(x)
    assert explanation.shape == (x.shape[0], x.shape[1])


def test_rlearner(uplift_data):
    x, t, y = uplift_data
    model = RLearner(
        x, t, y, mu_penalty=0.1, p_penalty=0.1, p_max_iter=10, tau_penalty=0.1
    )

    ite = model.predict_uplift(x)
    assert ite.shape == (x.shape[0],)

    explanation = model.explain_uplift(x)
    assert explanation.shape == (x.shape[0], x.shape[1])


def test_xlearner(uplift_data):
    x, t, y = uplift_data
    model = XLearner(
        x, t, y, mu_penalty=0.1, p_penalty=0.1, p_max_iter=10, tau_penalty=0.1
    )

    ite = model.predict_uplift(x)
    assert ite.shape == (x.shape[0],)

    explanation = model.explain_uplift(x)
    assert explanation.shape == (x.shape[0], x.shape[1])
