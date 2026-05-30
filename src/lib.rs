use pyo3::prelude::*;

pub mod metalearners;
pub mod python;
pub mod xmodels;

#[pymodule]
fn xuplift(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python::PyClassifier>()?;
    m.add_class::<python::PyRegressor>()?;

    m.add_class::<python::PyDRLearner>()?;
    m.add_class::<python::PyGRLearner>()?;
    m.add_class::<python::PyMLearner>()?;
    m.add_class::<python::PyPWLearner>()?;
    m.add_class::<python::PyRLearner>()?;
    m.add_class::<python::PySLearner>()?;
    m.add_class::<python::PyTLearner>()?;
    m.add_class::<python::PyXLearner>()?;

    Ok(())
}
