import numpy as np
import pytest
from xuplift import (
    DRRegressor,
    GRRegressor,
    MRegressor,
    PWRegressor,
    RRegressor,
    SRegressor,
    TRegressor,
    XRegressor,
)


@pytest.fixture
def uplift_data():
    np.random.seed(42)
    n_samples = 200
    n_features = 3
    x = np.random.randn(n_samples, n_features).astype(np.float32)
    t = np.random.randint(0, 2, n_samples).astype(np.float32)
    uplift = x[:, 0] * 2  # Uplift depends on x[0]

    # y = base_effect + t * uplift + noise
    y = (x[:, 1] + t * uplift + np.random.randn(n_samples) * 0.1).astype(np.float32)
    return x, t, y


def test_drregressor(uplift_data):
    x, t, y = uplift_data
    model = DRRegressor(
        x,
        t,
        y,
        [False, False, False],
        max_bases=64,
        mu_penalty=0.1,
        p_penalty=0.1,
        p_max_iter=10,
        tau_penalty=0.1,
    )

    ite = model.predict_uplift(x)
    assert ite.shape == (x.shape[0],)

    explanation = model.explain_uplift(x)
    assert explanation.shape == (x.shape[0], x.shape[1])


def test_grregressor(uplift_data):
    x, t, y = uplift_data
    model = GRRegressor(
        x,
        t,
        y,
        [False, False, False],
        max_bases=64,
        mu_penalty=0.1,
        p_penalty=0.1,
        tau_penalty=0.1,
    )

    ite = model.predict_uplift(x)
    assert ite.shape == (x.shape[0],)

    explanation = model.explain_uplift(x)
    assert explanation.shape == (x.shape[0], x.shape[1])


def test_mregressor(uplift_data):
    x, t, y = uplift_data
    model = MRegressor(x, t, y, [False, False, False], max_bases=64, tau_penalty=0.1)

    ite = model.predict_uplift(x)
    assert ite.shape == (x.shape[0],)

    explanation = model.explain_uplift(x)
    assert explanation.shape == (x.shape[0], x.shape[1])


def test_pwregressor(uplift_data):
    x, t, y = uplift_data
    model = PWRegressor(
        x,
        t,
        y,
        [False, False, False],
        max_bases=64,
        p_penalty=0.1,
        p_max_iter=10,
        tau_penalty=0.1,
    )

    ite = model.predict_uplift(x)
    assert ite.shape == (x.shape[0],)

    explanation = model.explain_uplift(x)
    assert explanation.shape == (x.shape[0], x.shape[1])


def test_rregressor(uplift_data):
    x, t, y = uplift_data
    model = RRegressor(
        x,
        t,
        y,
        [False, False, False],
        max_bases=64,
        mu_penalty=0.1,
        p_penalty=0.1,
        p_max_iter=10,
        tau_penalty=0.1,
    )

    ite = model.predict_uplift(x)
    assert ite.shape == (x.shape[0],)

    explanation = model.explain_uplift(x)
    assert explanation.shape == (x.shape[0], x.shape[1])


def test_sregressor(uplift_data):
    x, t, y = uplift_data
    model = SRegressor(x, t, y, [False, False, False], max_bases=64, mu_penalty=0.1)

    ite = model.predict_uplift(x)
    assert ite.shape == (x.shape[0],)

    explanation = model.explain_uplift(x)
    # SRegressor explain_uplift returns (n_samples, n_features + 1)
    assert explanation.shape == (x.shape[0], x.shape[1] + 1)


def test_tregressor(uplift_data):
    x, t, y = uplift_data
    model = TRegressor(x, t, y, [False, False, False], max_bases=64, mu_penalty=0.1)

    ite = model.predict_uplift(x)
    assert ite.shape == (x.shape[0],)

    explanation = model.explain_uplift(x)
    assert explanation.shape == (x.shape[0], x.shape[1])


def test_xregressor(uplift_data):
    x, t, y = uplift_data
    model = XRegressor(
        x,
        t,
        y,
        [False, False, False],
        max_bases=64,
        mu_penalty=0.1,
        p_penalty=0.1,
        p_max_iter=10,
        tau_penalty=0.1,
    )

    ite = model.predict_uplift(x)
    assert ite.shape == (x.shape[0],)

    explanation = model.explain_uplift(x)
    assert explanation.shape == (x.shape[0], x.shape[1])
