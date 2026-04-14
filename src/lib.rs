use pyo3::prelude::*;

pub mod feature_map;
pub mod metalearners;
pub mod python;
pub mod xmodels;

pub use crate::feature_map::KernelFeatureMap;
pub use crate::metalearners::rlearner::RLearner;
pub use crate::metalearners::slearner::SLearner;
pub use crate::metalearners::tlearner::TLearner;
pub use crate::metalearners::xlearner::XLearner;
pub use crate::xmodels::classifier::Classifier;
pub use crate::xmodels::regressor::Regressor;

#[pymodule]
fn xuplift(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python::PyClassifier>()?;
    m.add_class::<python::PyRegressor>()?;
    m.add_class::<python::PyRLearner>()?;
    m.add_class::<python::PySLearner>()?;
    m.add_class::<python::PyTLearner>()?;
    m.add_class::<python::PyXLearner>()?;

    Ok(())
}
