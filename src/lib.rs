#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

pub mod metalearners;
pub mod python;
pub mod xmodels;

#[pymodule]
fn xuplift(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python::PyClassifier>()?;
    m.add_class::<python::PyRegressor>()?;

    m.add_class::<python::PyDRClassifier>()?;
    m.add_class::<python::PyDRRegressor>()?;

    m.add_class::<python::PyGRClassifier>()?;
    m.add_class::<python::PyGRRegressor>()?;

    m.add_class::<python::PyMRegressor>()?;

    m.add_class::<python::PyPWRegressor>()?;

    m.add_class::<python::PyRClassifier>()?;
    m.add_class::<python::PyRRegressor>()?;

    m.add_class::<python::PySClassifier>()?;
    m.add_class::<python::PySRegressor>()?;

    m.add_class::<python::PyTClassifier>()?;
    m.add_class::<python::PyTRegressor>()?;

    m.add_class::<python::PyXClassifier>()?;
    m.add_class::<python::PyXRegressor>()?;

    Ok(())
}
