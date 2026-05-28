import numpy as np
from numpy.typing import NDArray

class XModel:
    def fit(self, x: NDArray[np.float32], y: NDArray[np.float32]) -> None: ...
    def predict(self, x: NDArray[np.float32]) -> NDArray[np.float32]: ...
    def explain(self, x: NDArray[np.float32]) -> NDArray[np.float32]: ...

class Classifier(XModel):
    def __init__(self, penalty: float, max_iter: int) -> None: ...

class Regressor(XModel):
    def __init__(self, penalty: float) -> None: ...

class Learner:
    def predict_uplift(self, x: NDArray[np.float32]) -> NDArray[np.float32]: ...
    def explain_uplift(self, x: NDArray[np.float32]) -> NDArray[np.float32]: ...

class DRLearner(Learner):
    def __init__(
        self,
        x: NDArray[np.float32],
        t: NDArray[np.float32],
        y: NDArray[np.float32],
        mu_penalty: float,
        p_penalty: float,
        p_max_iter: int,
        tau_penalty: float,
    ) -> None: ...

class GRLearner(Learner):
    def __init__(
        self,
        x: NDArray[np.float32],
        t: NDArray[np.float32],
        y: NDArray[np.float32],
        mu_penalty: float,
        p_penalty: float,
        tau_penalty: float,
    ) -> None: ...

class MLearner(Learner):
    def __init__(
        self,
        x: NDArray[np.float32],
        t: NDArray[np.float32],
        y: NDArray[np.float32],
        tau_penalty: float,
    ) -> None: ...

class PWLearner(Learner):
    def __init__(
        self,
        x: NDArray[np.float32],
        t: NDArray[np.float32],
        y: NDArray[np.float32],
        p_penalty: float,
        p_max_iter: int,
        tau_penalty: float,
    ) -> None: ...

class RLearner(Learner):
    def __init__(
        self,
        x: NDArray[np.float32],
        t: NDArray[np.float32],
        y: NDArray[np.float32],
        mu_penalty: float,
        p_penalty: float,
        p_max_iter: int,
        tau_penalty: float,
    ) -> None: ...

class SLearner(Learner):
    def __init__(
        self,
        x: NDArray[np.float32],
        t: NDArray[np.float32],
        y: NDArray[np.float32],
        mu_penalty: float,
    ) -> None: ...

class TLearner(Learner):
    def __init__(
        self,
        x: NDArray[np.float32],
        t: NDArray[np.float32],
        y: NDArray[np.float32],
        mu_penalty: float,
    ) -> None: ...

class XLearner(Learner):
    def __init__(
        self,
        x: NDArray[np.float32],
        t: NDArray[np.float32],
        y: NDArray[np.float32],
        mu_penalty: float,
        p_penalty: float,
        p_max_iter: int,
        tau_penalty: float,
    ) -> None: ...
