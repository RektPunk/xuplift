import numpy as np
import pytest
from xuplift import (
    DRClassifier,
    DRRegressor,
    GRClassifier,
    GRRegressor,
    MRegressor,
    PWRegressor,
    RClassifier,
    RRegressor,
    SClassifier,
    SRegressor,
    TClassifier,
    TRegressor,
    XClassifier,
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


def test_drlearner(uplift_data):
    x, t, y = uplift_data
    regressor = DRRegressor(
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

    ite = regressor.predict_uplift(x)
    assert ite.shape == (x.shape[0],)

    explanation = regressor.explain_uplift(x)
    assert explanation.shape == (x.shape[0], x.shape[1])

    classifier = DRClassifier(
        x,
        t,
        y,
        [False, False, False],
        max_bases=64,
        mu_penalty=0.1,
        mu_max_iter=10,
        p_penalty=0.1,
        p_max_iter=10,
        tau_penalty=0.1,
    )

    ite = classifier.predict_uplift(x)
    assert ite.shape == (x.shape[0],)

    explanation = classifier.explain_uplift(x)
    assert explanation.shape == (x.shape[0], x.shape[1])


def test_grlearner(uplift_data):
    x, t, y = uplift_data
    regressor = GRRegressor(
        x,
        t,
        y,
        [False, False, False],
        max_bases=64,
        mu_penalty=0.1,
        p_penalty=0.1,
        tau_penalty=0.1,
    )

    ite = regressor.predict_uplift(x)
    assert ite.shape == (x.shape[0],)

    explanation = regressor.explain_uplift(x)
    assert explanation.shape == (x.shape[0], x.shape[1])

    classifier = GRClassifier(
        x,
        t,
        y,
        [False, False, False],
        max_bases=64,
        mu_penalty=0.1,
        mu_max_iter=10,
        p_penalty=0.1,
        tau_penalty=0.1,
    )

    ite = classifier.predict_uplift(x)
    assert ite.shape == (x.shape[0],)

    explanation = classifier.explain_uplift(x)
    assert explanation.shape == (x.shape[0], x.shape[1])


def test_mlearner(uplift_data):
    x, t, y = uplift_data
    regressor = MRegressor(
        x, t, y, [False, False, False], max_bases=64, tau_penalty=0.1
    )

    ite = regressor.predict_uplift(x)
    assert ite.shape == (x.shape[0],)

    explanation = regressor.explain_uplift(x)
    assert explanation.shape == (x.shape[0], x.shape[1])


def test_pwlearner(uplift_data):
    x, t, y = uplift_data
    regressor = PWRegressor(
        x,
        t,
        y,
        [False, False, False],
        max_bases=64,
        p_penalty=0.1,
        p_max_iter=10,
        tau_penalty=0.1,
    )

    ite = regressor.predict_uplift(x)
    assert ite.shape == (x.shape[0],)

    explanation = regressor.explain_uplift(x)
    assert explanation.shape == (x.shape[0], x.shape[1])


def test_rlearner(uplift_data):
    x, t, y = uplift_data
    regressor = RRegressor(
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

    ite = regressor.predict_uplift(x)
    assert ite.shape == (x.shape[0],)

    explanation = regressor.explain_uplift(x)
    assert explanation.shape == (x.shape[0], x.shape[1])

    classifier = RClassifier(
        x,
        t,
        y,
        [False, False, False],
        max_bases=64,
        mu_penalty=0.1,
        mu_max_iter=10,
        p_penalty=0.1,
        p_max_iter=10,
        tau_penalty=0.1,
    )

    ite = classifier.predict_uplift(x)
    assert ite.shape == (x.shape[0],)

    explanation = classifier.explain_uplift(x)
    assert explanation.shape == (x.shape[0], x.shape[1])


def test_slearner(uplift_data):
    x, t, y = uplift_data
    regressor = SRegressor(x, t, y, [False, False, False], max_bases=64, mu_penalty=0.1)

    ite = regressor.predict_uplift(x)
    assert ite.shape == (x.shape[0],)

    explanation = regressor.explain_uplift(x)
    # SRegressor explain_uplift returns (n_samples, n_features + 1)
    assert explanation.shape == (x.shape[0], x.shape[1] + 1)

    classifier = SClassifier(
        x, t, y, [False, False, False], max_bases=64, mu_penalty=0.1, mu_max_iter=10
    )

    ite = classifier.predict_uplift(x)
    assert ite.shape == (x.shape[0],)

    explanation = classifier.explain_uplift(x)
    assert explanation.shape == (x.shape[0], x.shape[1] + 1)


def test_tlearner(uplift_data):
    x, t, y = uplift_data
    regressor = TRegressor(x, t, y, [False, False, False], max_bases=64, mu_penalty=0.1)

    ite = regressor.predict_uplift(x)
    assert ite.shape == (x.shape[0],)

    explanation = regressor.explain_uplift(x)
    assert explanation.shape == (x.shape[0], x.shape[1])

    classifier = TClassifier(
        x, t, y, [False, False, False], max_bases=64, mu_penalty=0.1, mu_max_iter=10
    )

    ite = classifier.predict_uplift(x)
    assert ite.shape == (x.shape[0],)

    explanation = classifier.explain_uplift(x)
    assert explanation.shape == (x.shape[0], x.shape[1])


def test_xlearner(uplift_data):
    x, t, y = uplift_data
    regressor = XRegressor(
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

    ite = regressor.predict_uplift(x)
    assert ite.shape == (x.shape[0],)

    explanation = regressor.explain_uplift(x)
    assert explanation.shape == (x.shape[0], x.shape[1])

    classifier = XClassifier(
        x,
        t,
        y,
        [False, False, False],
        max_bases=64,
        mu_penalty=0.1,
        mu_max_iter=10,
        p_penalty=0.1,
        p_max_iter=10,
        tau_penalty=0.1,
    )

    ite = classifier.predict_uplift(x)
    assert ite.shape == (x.shape[0],)

    explanation = classifier.explain_uplift(x)
    assert explanation.shape == (x.shape[0], x.shape[1])
