use faer::{Col, ColRef, Mat, MatRef};
use numpy::ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

pub use crate::feature_map::KernelFeatureMap;
pub use crate::xmodels::classifier::Classifier;
pub use crate::xmodels::regressor::Regressor;

pub use crate::metalearners::rlearner::RLearner;
pub use crate::metalearners::slearner::SLearner;
pub use crate::metalearners::tlearner::TLearner;
pub use crate::metalearners::xlearner::XLearner;

fn convert_to_faer_mat(x: PyReadonlyArray2<'_, f32>) -> MatRef<'_, f32> {
    let raw_arr = x.as_raw_array();
    let nrows = raw_arr.nrows();
    let ncols = raw_arr.ncols();
    let strides: [isize; 2] = raw_arr.strides().try_into().unwrap();
    unsafe { MatRef::from_raw_parts(raw_arr.as_ptr(), nrows, ncols, strides[0], strides[1]) }
}

fn convert_to_faer_col(x: PyReadonlyArray1<'_, f32>) -> ColRef<'_, f32> {
    let raw_arr = x.as_raw_array();
    let nrows = raw_arr.len();
    let strides: [isize; 1] = raw_arr.strides().try_into().unwrap();
    unsafe { ColRef::from_raw_parts(raw_arr.as_ptr(), nrows, strides[0]) }
}

fn convert_to_numpy_mat(x: Mat<f32>) -> Array2<f32> {
    Array2::from_shape_fn((x.nrows(), x.ncols()), |(i, j)| x[(i, j)])
}

fn convert_to_numpy_col(x: Col<f32>) -> Array1<f32> {
    Array1::from_iter(x.iter().copied())
}

fn prepare_input<'a>(
    x: PyReadonlyArray2<'a, f32>,
    t: PyReadonlyArray1<'a, f32>,
    y: PyReadonlyArray1<'a, f32>,
) -> (MatRef<'a, f32>, ColRef<'a, f32>, ColRef<'a, f32>) {
    let x_mat = convert_to_faer_mat(x);
    let t_col = convert_to_faer_col(t);
    let y_col = convert_to_faer_col(y);
    (x_mat, t_col, y_col)
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
        let x_mat = convert_to_faer_mat(x);
        let y_col = convert_to_faer_col(y);
        self.inner.fit(x_mat, y_col);
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let x_mat = convert_to_faer_mat(x);
        let pred = self.inner.predict(x_mat);
        let py_pred = convert_to_numpy_col(pred).to_pyarray(py);
        Ok(py_pred)
    }

    fn explain<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let x_mat = convert_to_faer_mat(x);
        let explanation = self.inner.explain(x_mat);
        let py_expl = convert_to_numpy_mat(explanation).to_pyarray(py);
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
        let x_mat = convert_to_faer_mat(x);
        let y_col = convert_to_faer_col(y);
        self.inner.fit(x_mat, y_col);
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let x_mat = convert_to_faer_mat(x);
        let pred = self.inner.predict(x_mat);
        let py_pred = convert_to_numpy_col(pred).to_pyarray(py);
        Ok(py_pred)
    }

    fn explain<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let x_mat = convert_to_faer_mat(x);
        let explanation = self.inner.explain(x_mat);
        let py_expl = convert_to_numpy_mat(explanation).to_pyarray(py);
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
        let x_mat = convert_to_faer_mat(x);
        let uplift = self.inner.predict_uplift(x_mat);
        let py_pred = convert_to_numpy_col(uplift).to_pyarray(py);
        Ok(py_pred)
    }

    fn explain_uplift<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let x_mat = convert_to_faer_mat(x);
        let explanation = self.inner.explain_uplift(x_mat);
        let py_expl = convert_to_numpy_mat(explanation).to_pyarray(py);
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
        let x_mat = convert_to_faer_mat(x);
        let uplift = self.inner.predict_uplift(x_mat);
        let py_pred = convert_to_numpy_col(uplift).to_pyarray(py);
        Ok(py_pred)
    }

    fn explain_uplift<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let x_mat = convert_to_faer_mat(x);
        let explanation = self.inner.explain_uplift(x_mat);
        let py_expl = convert_to_numpy_mat(explanation).to_pyarray(py);
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
        let x_mat = convert_to_faer_mat(x);
        let uplift = self.inner.predict_uplift(x_mat);
        let py_pred = convert_to_numpy_col(uplift).to_pyarray(py);
        Ok(py_pred)
    }

    fn explain_uplift<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let x_mat = convert_to_faer_mat(x);
        let explanation = self.inner.explain_uplift(x_mat);
        let py_expl = convert_to_numpy_mat(explanation).to_pyarray(py);
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
        let x_mat = convert_to_faer_mat(x);
        let uplift = self.inner.predict_uplift(x_mat);
        let py_pred = convert_to_numpy_col(uplift).to_pyarray(py);
        Ok(py_pred)
    }

    fn explain_uplift<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let x_mat = convert_to_faer_mat(x);
        let explanation = self.inner.explain_uplift(x_mat);
        let py_expl = convert_to_numpy_mat(explanation).to_pyarray(py);
        Ok(py_expl)
    }
}
