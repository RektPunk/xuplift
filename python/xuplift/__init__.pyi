import numpy as np
from numpy.typing import NDArray

class XModel:
    def fit(
        self, x: NDArray[np.float32], y: NDArray[np.float32], is_categorical: list[bool]
    ) -> None: ...
    def predict(self, x: NDArray[np.float32]) -> NDArray[np.float32]: ...
    def explain(self, x: NDArray[np.float32]) -> NDArray[np.float32]: ...

class Classifier(XModel):
    def __init__(self, max_bases: int, penalty: float, max_iter: int) -> None: ...

class Regressor(XModel):
    def __init__(self, max_bases: int, penalty: float) -> None: ...

class MetaLearner:
    def predict_uplift(self, x: NDArray[np.float32]) -> NDArray[np.float32]: ...
    def explain_uplift(self, x: NDArray[np.float32]) -> NDArray[np.float32]: ...

class DRClassifier(MetaLearner):
    def __init__(
        self,
        x: NDArray[np.float32],
        t: NDArray[np.float32],
        y: NDArray[np.float32],
        is_categorical: list[bool],
        max_bases: int,
        mu_penalty: float,
        mu_max_iter: int,
        p_penalty: float,
        p_max_iter: int,
        tau_penalty: float,
    ) -> None: ...

class DRRegressor(MetaLearner):
    def __init__(
        self,
        x: NDArray[np.float32],
        t: NDArray[np.float32],
        y: NDArray[np.float32],
        is_categorical: list[bool],
        max_bases: int,
        mu_penalty: float,
        p_penalty: float,
        p_max_iter: int,
        tau_penalty: float,
    ) -> None: ...

class GRClassifier(MetaLearner):
    def __init__(
        self,
        x: NDArray[np.float32],
        t: NDArray[np.float32],
        y: NDArray[np.float32],
        is_categorical: list[bool],
        max_bases: int,
        mu_penalty: float,
        mu_max_iter: int,
        p_penalty: float,
        tau_penalty: float,
    ) -> None: ...

class GRRegressor(MetaLearner):
    def __init__(
        self,
        x: NDArray[np.float32],
        t: NDArray[np.float32],
        y: NDArray[np.float32],
        is_categorical: list[bool],
        max_bases: int,
        mu_penalty: float,
        p_penalty: float,
        tau_penalty: float,
    ) -> None: ...

class MRegressor(MetaLearner):
    def __init__(
        self,
        x: NDArray[np.float32],
        t: NDArray[np.float32],
        y: NDArray[np.float32],
        is_categorical: list[bool],
        max_bases: int,
        tau_penalty: float,
    ) -> None: ...

class PWRegressor(MetaLearner):
    def __init__(
        self,
        x: NDArray[np.float32],
        t: NDArray[np.float32],
        y: NDArray[np.float32],
        is_categorical: list[bool],
        max_bases: int,
        p_penalty: float,
        p_max_iter: int,
        tau_penalty: float,
    ) -> None: ...

class RClassifier(MetaLearner):
    def __init__(
        self,
        x: NDArray[np.float32],
        t: NDArray[np.float32],
        y: NDArray[np.float32],
        is_categorical: list[bool],
        max_bases: int,
        mu_penalty: float,
        mu_max_iter: int,
        p_penalty: float,
        p_max_iter: int,
        tau_penalty: float,
    ) -> None: ...

class RRegressor(MetaLearner):
    def __init__(
        self,
        x: NDArray[np.float32],
        t: NDArray[np.float32],
        y: NDArray[np.float32],
        is_categorical: list[bool],
        max_bases: int,
        mu_penalty: float,
        p_penalty: float,
        p_max_iter: int,
        tau_penalty: float,
    ) -> None: ...

class SClassifier(MetaLearner):
    def __init__(
        self,
        x: NDArray[np.float32],
        t: NDArray[np.float32],
        y: NDArray[np.float32],
        is_categorical: list[bool],
        max_bases: int,
        mu_penalty: float,
        mu_max_iter: int,
    ) -> None: ...

class SRegressor(MetaLearner):
    def __init__(
        self,
        x: NDArray[np.float32],
        t: NDArray[np.float32],
        y: NDArray[np.float32],
        is_categorical: list[bool],
        max_bases: int,
        mu_penalty: float,
    ) -> None: ...

class TClassifier(MetaLearner):
    def __init__(
        self,
        x: NDArray[np.float32],
        t: NDArray[np.float32],
        y: NDArray[np.float32],
        is_categorical: list[bool],
        max_bases: int,
        mu_penalty: float,
        mu_max_iter: int,
    ) -> None: ...

class TRegressor(MetaLearner):
    def __init__(
        self,
        x: NDArray[np.float32],
        t: NDArray[np.float32],
        y: NDArray[np.float32],
        is_categorical: list[bool],
        max_bases: int,
        mu_penalty: float,
    ) -> None: ...

class XClassifier(MetaLearner):
    def __init__(
        self,
        x: NDArray[np.float32],
        t: NDArray[np.float32],
        y: NDArray[np.float32],
        is_categorical: list[bool],
        max_bases: int,
        mu_penalty: float,
        mu_max_iter: int,
        p_penalty: float,
        p_max_iter: int,
        tau_penalty: float,
    ) -> None: ...

class XRegressor(MetaLearner):
    def __init__(
        self,
        x: NDArray[np.float32],
        t: NDArray[np.float32],
        y: NDArray[np.float32],
        is_categorical: list[bool],
        max_bases: int,
        mu_penalty: float,
        p_penalty: float,
        p_max_iter: int,
        tau_penalty: float,
    ) -> None: ...
