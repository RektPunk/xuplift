use faer::{ColRef, MatRef};
use numpy::ndarray::{ArrayView1, ArrayView2, ShapeBuilder};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

pub use crate::feature_map::KernelFeatureMap;
pub use crate::xmodels::classifier::Classifier;
pub use crate::xmodels::regressor::Regressor;

pub use crate::metalearners::drlearner::DRLearner;
pub use crate::metalearners::rlearner::RLearner;
pub use crate::metalearners::slearner::SLearner;
pub use crate::metalearners::tlearner::TLearner;
pub use crate::metalearners::xlearner::XLearner;

/// Converts `numpy` views into `faer` reference types.
pub trait IntoFaer {
    type Faer;
    fn into_faer(self) -> Self::Faer;
}

/// Converts `faer` types into `ndarray` views.
pub trait IntoNdarray {
    type Ndarray;
    fn into_ndarray(self) -> Self::Ndarray;
}

// Implementations for `numpy` -> `faer` conversions
impl<'a, T: numpy::Element + 'a> IntoFaer for PyReadonlyArray2<'a, T> {
    type Faer = MatRef<'a, T>;

    /// Converts a `PyReadonlyArray2` into a `faer::MatRef`.
    fn into_faer(self) -> Self::Faer {
        let raw_arr = self.as_raw_array();
        let nrows = raw_arr.nrows();
        let ncols = raw_arr.ncols();
        let strides: [isize; 2] = raw_arr.strides().try_into().unwrap();
        unsafe { MatRef::from_raw_parts(raw_arr.as_ptr(), nrows, ncols, strides[0], strides[1]) }
    }
}

impl<'a, T: numpy::Element + 'a> IntoFaer for PyReadonlyArray1<'a, T> {
    type Faer = ColRef<'a, T>;

    /// Converts a `PyReadonlyArray1` into a `faer::ColRef`.
    fn into_faer(self) -> Self::Faer {
        let raw_arr = self.as_raw_array();
        let nrows = raw_arr.len();
        let strides: [isize; 1] = raw_arr.strides().try_into().unwrap();
        unsafe { ColRef::from_raw_parts(raw_arr.as_ptr(), nrows, strides[0]) }
    }
}

// Implementations for `faer` -> `numpy` conversions
impl<'a, T> IntoNdarray for MatRef<'a, T> {
    type Ndarray = ArrayView2<'a, T>;

    /// Converts a `faer::MatRef` into an `ndarray::ArrayView2`.
    fn into_ndarray(self) -> Self::Ndarray {
        let nrows = self.nrows();
        let ncols = self.ncols();
        let row_stride = self.row_stride() as usize;
        let col_stride = self.col_stride() as usize;
        unsafe {
            ArrayView2::from_shape_ptr(
                (nrows, ncols).strides((row_stride, col_stride)),
                self.as_ptr(),
            )
        }
    }
}

impl<'a, T> IntoNdarray for ColRef<'a, T> {
    type Ndarray = ArrayView1<'a, T>;

    /// Converts a `faer::ColRef` into an `ndarray::ArrayView1`.
    fn into_ndarray(self) -> Self::Ndarray {
        let nrows = self.nrows();
        let row_stride = self.row_stride() as usize;
        unsafe { ArrayView1::from_shape_ptr(nrows.strides(row_stride), self.as_ptr()) }
    }
}

/// Helper function to prepare multiple inputs (X, T, Y) from Python into `faer` types.
fn prepare_input<'a>(
    x: PyReadonlyArray2<'a, f32>,
    t: PyReadonlyArray1<'a, f32>,
    y: PyReadonlyArray1<'a, f32>,
) -> (MatRef<'a, f32>, ColRef<'a, f32>, ColRef<'a, f32>) {
    (x.into_faer(), t.into_faer(), y.into_faer())
}

#[pyclass(name = "Classifier")]
pub struct PyClassifier {
    inner: Classifier,
}
#[pymethods]
impl PyClassifier {
    #[new]
    fn new(penalty: f32, max_iter: usize) -> Self {
        let classifier = Classifier::new(penalty, max_iter);
        PyClassifier { inner: classifier }
    }

    fn fit(&mut self, x: PyReadonlyArray2<f32>, y: PyReadonlyArray1<f32>) {
        self.inner.fit(x.into_faer(), y.into_faer());
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let pred = self.inner.predict(x.into_faer());
        let py_pred = pred.as_ref().into_ndarray().to_pyarray(py);
        Ok(py_pred)
    }

    fn explain<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let explanation = self.inner.explain(x.into_faer());
        let py_expl = explanation.as_ref().into_ndarray().to_pyarray(py);
        Ok(py_expl)
    }
}

#[pyclass(name = "Regressor")]
pub struct PyRegressor {
    inner: Regressor,
}
#[pymethods]
impl PyRegressor {
    #[new]
    fn new(penalty: f32) -> Self {
        let regressor = Regressor::new(penalty);
        PyRegressor { inner: regressor }
    }

    fn fit(&mut self, x: PyReadonlyArray2<f32>, y: PyReadonlyArray1<f32>) {
        self.inner.fit(x.into_faer(), y.into_faer());
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let pred = self.inner.predict(x.into_faer());
        let py_pred = pred.as_ref().into_ndarray().to_pyarray(py);
        Ok(py_pred)
    }

    fn explain<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let explanation = self.inner.explain(x.into_faer());
        let py_expl = explanation.as_ref().into_ndarray().to_pyarray(py);
        Ok(py_expl)
    }
}

#[pyclass(name = "DRLearner")]
pub struct PyDRLearner {
    inner: DRLearner,
}
#[pymethods]
impl PyDRLearner {
    #[new]
    fn new(
        x: PyReadonlyArray2<f32>,
        t: PyReadonlyArray1<f32>,
        y: PyReadonlyArray1<f32>,
        mu_penalty: f32,
        p_penalty: f32,
        p_max_iter: usize,
        tau_penalty: f32,
    ) -> Self {
        let (x_mat, t_col, y_col) = prepare_input(x, t, y);
        let model = DRLearner::new(
            x_mat,
            t_col,
            y_col,
            mu_penalty,
            p_penalty,
            p_max_iter,
            tau_penalty,
        );
        PyDRLearner { inner: model }
    }

    fn predict_uplift<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let uplift = self.inner.predict_uplift(x.into_faer());
        let py_pred = uplift.as_ref().into_ndarray().to_pyarray(py);
        Ok(py_pred)
    }

    fn explain_uplift<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let explanation = self.inner.explain_uplift(x.into_faer());
        let py_expl = explanation.as_ref().into_ndarray().to_pyarray(py);
        Ok(py_expl)
    }
}

#[pyclass(name = "RLearner")]
pub struct PyRLearner {
    inner: RLearner,
}
#[pymethods]
impl PyRLearner {
    #[new]
    fn new(
        x: PyReadonlyArray2<f32>,
        t: PyReadonlyArray1<f32>,
        y: PyReadonlyArray1<f32>,
        mu_penalty: f32,
        p_penalty: f32,
        p_max_iter: usize,
        tau_penalty: f32,
    ) -> Self {
        let (x_mat, t_col, y_col) = prepare_input(x, t, y);
        let model = RLearner::new(
            x_mat,
            t_col,
            y_col,
            mu_penalty,
            p_penalty,
            p_max_iter,
            tau_penalty,
        );
        PyRLearner { inner: model }
    }

    fn predict_uplift<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let uplift = self.inner.predict_uplift(x.into_faer());
        let py_pred = uplift.as_ref().into_ndarray().to_pyarray(py);
        Ok(py_pred)
    }

    fn explain_uplift<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let explanation = self.inner.explain_uplift(x.into_faer());
        let py_expl = explanation.as_ref().into_ndarray().to_pyarray(py);
        Ok(py_expl)
    }
}

#[pyclass(name = "SLearner")]
pub struct PySLearner {
    inner: SLearner,
}
#[pymethods]
impl PySLearner {
    #[new]
    fn new(
        x: PyReadonlyArray2<f32>,
        t: PyReadonlyArray1<f32>,
        y: PyReadonlyArray1<f32>,
        mu_penalty: f32,
    ) -> Self {
        let (x_mat, t_col, y_col) = prepare_input(x, t, y);
        let model = SLearner::new(x_mat, t_col, y_col, mu_penalty);
        PySLearner { inner: model }
    }

    fn predict_uplift<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let uplift = self.inner.predict_uplift(x.into_faer());
        let py_pred = uplift.as_ref().into_ndarray().to_pyarray(py);
        Ok(py_pred)
    }

    fn explain_uplift<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let explanation = self.inner.explain_uplift(x.into_faer());
        let py_expl = explanation.as_ref().into_ndarray().to_pyarray(py);
        Ok(py_expl)
    }
}

#[pyclass(name = "TLearner")]
pub struct PyTLearner {
    inner: TLearner,
}
#[pymethods]
impl PyTLearner {
    #[new]
    fn new(
        x: PyReadonlyArray2<f32>,
        t: PyReadonlyArray1<f32>,
        y: PyReadonlyArray1<f32>,
        mu_penalty: f32,
    ) -> Self {
        let (x_mat, t_col, y_col) = prepare_input(x, t, y);
        let model = TLearner::new(x_mat, t_col, y_col, mu_penalty);
        PyTLearner { inner: model }
    }

    fn predict_uplift<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let uplift = self.inner.predict_uplift(x.into_faer());
        let py_pred = uplift.as_ref().into_ndarray().to_pyarray(py);
        Ok(py_pred)
    }

    fn explain_uplift<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let explanation = self.inner.explain_uplift(x.into_faer());
        let py_expl = explanation.as_ref().into_ndarray().to_pyarray(py);
        Ok(py_expl)
    }
}

#[pyclass(name = "XLearner")]
pub struct PyXLearner {
    inner: XLearner,
}
#[pymethods]
impl PyXLearner {
    #[new]
    fn new(
        x: PyReadonlyArray2<f32>,
        t: PyReadonlyArray1<f32>,
        y: PyReadonlyArray1<f32>,
        mu_penalty: f32,
        p_penalty: f32,
        p_max_iter: usize,
        tau_penalty: f32,
    ) -> Self {
        let (x_mat, t_col, y_col) = prepare_input(x, t, y);
        let model = XLearner::new(
            x_mat,
            t_col,
            y_col,
            mu_penalty,
            p_penalty,
            p_max_iter,
            tau_penalty,
        );
        PyXLearner { inner: model }
    }

    fn predict_uplift<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let uplift = self.inner.predict_uplift(x.into_faer());
        let py_pred = uplift.as_ref().into_ndarray().to_pyarray(py);
        Ok(py_pred)
    }

    fn explain_uplift<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let explanation = self.inner.explain_uplift(x.into_faer());
        let py_expl = explanation.as_ref().into_ndarray().to_pyarray(py);
        Ok(py_expl)
    }
}
