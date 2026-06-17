use faer::{ColRef, MatRef};
use numpy::ndarray::{ArrayView1, ArrayView2, ShapeBuilder};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

pub use crate::xmodels::classifier::Classifier;
pub use crate::xmodels::regressor::Regressor;

pub use crate::metalearners::drlearner::DRLearner;
pub use crate::metalearners::grlearner::GRLearner;
pub use crate::metalearners::mlearner::MLearner;
pub use crate::metalearners::pwlearner::PWLearner;
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

/// Macro to generate Python bindings for kernel based machine learning models.
///
/// This macro automates the boilerplate for creating the Python class wrapper,
/// the constructor with dynamic arguments, and standard methods such as `fit`, `predict`, and `explain`.
macro_rules! impl_py_xmodels {
    (
        $py_name:literal,
        $struct_name:ident,
        $inner_name:ident,
        ( $($args:ident : $types:ty),* ),
        ( $($pass_args:ident),* )
    ) => {
        #[pyclass(name = $py_name)]
        pub struct $struct_name {
            inner: $inner_name,
        }

        #[pymethods]
        impl $struct_name {
            #[new]
            fn new($($args : $types),*) -> Self {
                let model = $inner_name::new($($pass_args),*);
                Self { inner: model }
            }

            fn fit(
                &mut self,
                x: PyReadonlyArray2<f32>,
                y: PyReadonlyArray1<f32>,
                is_categorical: Vec<bool>,

            ) {
                self.inner.fit(x.into_faer(), y.into_faer(), &is_categorical);
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
    };
}

/// Macro to generate Python bindings for Causal Inference Metalearners.
///
/// This macro boilerplate-generates the Python wrapper class,
/// handling inputs `x` (features), `t` (treatment), and `y` (outcome) along with learner-specific penalty parameters.
/// It exposes `predict_uplift` and `explain_uplift` to the Python runtime.
macro_rules! impl_py_learner {
    (
        $py_name:literal,
        $struct_name:ident,
        $inner_name:ident,
        ( $($args:ident : $types:ty),* ),
        ( $($pass_args:ident),* )
    ) => {
        #[pyclass(name = $py_name)]
        pub struct $struct_name {
            inner: $inner_name,
        }

        #[pymethods]
        impl $struct_name {
            #[new]
            fn new(
                x: PyReadonlyArray2<f32>,
                t: PyReadonlyArray1<f32>,
                y: PyReadonlyArray1<f32>,
                is_categorical: Vec<bool>,
                $($args : $types),*
            ) -> Self {
                let x_mat = x.into_faer();
                let t_col = t.into_faer();
                let y_col = y.into_faer();
                let model = $inner_name::new(
                    x_mat,
                    t_col,
                    y_col,
                    &is_categorical,
                    $($pass_args),*
                );
                Self { inner: model }
            }

            fn predict_uplift<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f32>) -> PyResult<Bound<'py, PyArray1<f32>>> {
                let uplift = self.inner.predict_uplift(x.into_faer());
                Ok(uplift.as_ref().into_ndarray().to_pyarray(py))
            }

            fn explain_uplift<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f32>) -> PyResult<Bound<'py, PyArray2<f32>>> {
                let explanation = self.inner.explain_uplift(x.into_faer());
                Ok(explanation.as_ref().into_ndarray().to_pyarray(py))
            }
        }
    };
}

// Generate Python bindings for kernel based predictive models
impl_py_xmodels!(
    "Classifier", PyClassifier, Classifier,
    (max_bases: usize, penalty: f32, max_iter: usize),
    (max_bases, penalty, max_iter)
);

impl_py_xmodels!(
    "Regressor", PyRegressor, Regressor,
    (max_bases: usize, penalty: f32),
    (max_bases, penalty)
);

// Generate Python bindings for causal metalearners
impl_py_learner!(
    "DRLearner", PyDRLearner, DRLearner,
    (max_bases: usize, mu_penalty: f32, p_penalty: f32, p_max_iter: usize, tau_penalty: f32),
    (max_bases, mu_penalty, p_penalty, p_max_iter, tau_penalty)
);

impl_py_learner!(
    "GRLearner", PyGRLearner, GRLearner,
    (max_bases: usize, mu_penalty: f32, p_penalty: f32, tau_penalty: f32),
    (max_bases, mu_penalty, p_penalty, tau_penalty)
);

impl_py_learner!(
    "MLearner", PyMLearner, MLearner,
    (max_bases: usize, tau_penalty: f32),
    (max_bases, tau_penalty)
);

impl_py_learner!(
    "PWLearner", PyPWLearner, PWLearner,
    (max_bases: usize, p_penalty: f32, p_max_iter: usize, tau_penalty: f32),
    (max_bases, p_penalty, p_max_iter, tau_penalty)
);

impl_py_learner!(
    "SLearner", PySLearner, SLearner,
    (max_bases: usize, mu_penalty: f32),
    (max_bases, mu_penalty)
);

impl_py_learner!(
    "RLearner", PyRLearner, RLearner,
    (max_bases: usize, mu_penalty: f32, p_penalty: f32, p_max_iter: usize, tau_penalty: f32),
    (max_bases, mu_penalty, p_penalty, p_max_iter, tau_penalty)
);

impl_py_learner!(
    "TLearner", PyTLearner, TLearner,
    (max_bases: usize, mu_penalty: f32),
    (max_bases, mu_penalty)
);

impl_py_learner!(
    "XLearner", PyXLearner, XLearner,
    (max_bases: usize, mu_penalty: f32, p_penalty: f32, p_max_iter: usize, tau_penalty: f32),
    (max_bases, mu_penalty, p_penalty, p_max_iter, tau_penalty)
);
