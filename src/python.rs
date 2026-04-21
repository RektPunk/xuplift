use std::sync::Arc;

use faer::{Col, Mat};
use numpy::ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

pub use crate::feature_map::KernelFeatureMap;
pub use crate::xmodels::classifier::Classifier;
pub use crate::xmodels::regressor::Regressor;

pub use crate::metalearners::rlearner::RLearner;
pub use crate::metalearners::slearner::SLearner;
pub use crate::metalearners::tlearner::TLearner;
pub use crate::metalearners::xlearner::XLearner;

fn convert_to_faer_mat(x: PyReadonlyArray2<f32>) -> Mat<f32> {
    let x_view = x.as_array();
    Mat::from_fn(x_view.nrows(), x_view.ncols(), |i, j| x_view[[i, j]])
}

fn convert_to_faer_col(x: PyReadonlyArray1<f32>) -> Col<f32> {
    let x_view = x.as_array();
    Col::from_fn(x_view.len(), |i| x_view[i])
}

fn convert_to_numpy_mat(x: Mat<f32>) -> Array2<f32> {
    Array2::from_shape_fn((x.nrows(), x.ncols()), |(i, j)| x[(i, j)])
}

fn convert_to_numpy_col(x: Col<f32>) -> Array1<f32> {
    Array1::from_iter(x.iter().copied())
}

fn prepare_input(
    x: PyReadonlyArray2<f32>,
    t: PyReadonlyArray1<f32>,
    y: PyReadonlyArray1<f32>,
) -> (Mat<f32>, Col<f32>, Col<f32>) {
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
    fn new(x: PyReadonlyArray2<f32>) -> Self {
        let x_mat = convert_to_faer_mat(x);
        let mut map_x = KernelFeatureMap::new();
        map_x.fit(&x_mat);
        let map_t1_arc = Arc::new(map_x);
        let classifier = Classifier::new(map_t1_arc);
        PyClassifier { inner: classifier }
    }

    fn fit(&mut self, y: PyReadonlyArray1<f32>, max_iter: usize) {
        let y_col = convert_to_faer_col(y);
        self.inner.fit(&y_col, max_iter);
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let x_mat = convert_to_faer_mat(x);
        let pred = self.inner.predict(&x_mat);
        let py_pred = convert_to_numpy_col(pred).to_pyarray(py);
        Ok(py_pred)
    }

    fn explain<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let x_mat = convert_to_faer_mat(x);
        let explanation = self.inner.explain(&x_mat);
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
    fn new(x: PyReadonlyArray2<f32>) -> Self {
        let x_mat = convert_to_faer_mat(x);
        let mut map_x = KernelFeatureMap::new();
        map_x.fit(&x_mat);
        let map_t1_arc = Arc::new(map_x);
        let regressor = Regressor::new(map_t1_arc);
        PyRegressor { inner: regressor }
    }

    fn fit(&mut self, y: PyReadonlyArray1<f32>) {
        let y_col = convert_to_faer_col(y);
        self.inner.fit(&y_col);
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let x_mat = convert_to_faer_mat(x);
        let pred = self.inner.predict(&x_mat);
        let py_pred = convert_to_numpy_col(pred).to_pyarray(py);
        Ok(py_pred)
    }

    fn explain<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let x_mat = convert_to_faer_mat(x);
        let explanation = self.inner.explain(&x_mat);
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
    fn new(x: PyReadonlyArray2<f32>, t: PyReadonlyArray1<f32>, y: PyReadonlyArray1<f32>) -> Self {
        let (x_mat, t_col, y_col) = prepare_input(x, t, y);
        let model = RLearner::new(&x_mat, &t_col, &y_col);
        PyRLearner { inner: model }
    }

    fn predict_uplift<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let x_mat = convert_to_faer_mat(x);
        let uplift = self.inner.predict_uplift(&x_mat);
        let py_pred = convert_to_numpy_col(uplift).to_pyarray(py);
        Ok(py_pred)
    }

    fn explain_uplift<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let x_mat = convert_to_faer_mat(x);
        let explanation = self.inner.explain_uplift(&x_mat);
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
    fn new(x: PyReadonlyArray2<f32>, t: PyReadonlyArray1<f32>, y: PyReadonlyArray1<f32>) -> Self {
        let (x_mat, t_col, y_col) = prepare_input(x, t, y);
        let model = SLearner::new(&x_mat, &t_col, &y_col);
        PySLearner { inner: model }
    }

    fn predict_uplift<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let x_mat = convert_to_faer_mat(x);
        let uplift = self.inner.predict_uplift(&x_mat);
        let py_pred = convert_to_numpy_col(uplift).to_pyarray(py);
        Ok(py_pred)
    }

    fn explain_uplift<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let x_mat = convert_to_faer_mat(x);
        let explanation = self.inner.explain_uplift(&x_mat);
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
    fn new(x: PyReadonlyArray2<f32>, t: PyReadonlyArray1<f32>, y: PyReadonlyArray1<f32>) -> Self {
        let (x_mat, t_col, y_col) = prepare_input(x, t, y);
        let model = TLearner::new(&x_mat, &t_col, &y_col);
        PyTLearner { inner: model }
    }

    fn predict_uplift<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let x_mat = convert_to_faer_mat(x);
        let uplift = self.inner.predict_uplift(&x_mat);
        let py_pred = convert_to_numpy_col(uplift).to_pyarray(py);
        Ok(py_pred)
    }

    fn explain_uplift<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let x_mat = convert_to_faer_mat(x);
        let explanation = self.inner.explain_uplift(&x_mat);
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
    fn new(x: PyReadonlyArray2<f32>, t: PyReadonlyArray1<f32>, y: PyReadonlyArray1<f32>) -> Self {
        let (x_mat, t_col, y_col) = prepare_input(x, t, y);
        let model = XLearner::new(&x_mat, &t_col, &y_col);
        PyXLearner { inner: model }
    }

    fn predict_uplift<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let x_mat = convert_to_faer_mat(x);
        let uplift = self.inner.predict_uplift(&x_mat);
        let py_pred = convert_to_numpy_col(uplift).to_pyarray(py);
        Ok(py_pred)
    }

    fn explain_uplift<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let x_mat = convert_to_faer_mat(x);
        let explanation = self.inner.explain_uplift(&x_mat);
        let py_expl = convert_to_numpy_mat(explanation).to_pyarray(py);
        Ok(py_expl)
    }
}
