use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[pyfunction]
fn run_tachometer_cli(py: Python<'_>, args: Vec<String>) -> PyResult<()> {
    py.allow_threads(move || tachometer_scraper::run_cli(args).map_err(|error| error.to_string()))
        .map_err(PyRuntimeError::new_err)
}

#[pymodule]
fn _tachometer(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(run_tachometer_cli, module)?)?;
    Ok(())
}
