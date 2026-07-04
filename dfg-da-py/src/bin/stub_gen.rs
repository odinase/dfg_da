//! Emits the `dfg_da_py.pyi` type stub next to the extension module so Python
//! linters/IDEs see the signatures. Run after building: `cargo run --bin stub_gen`.

use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    let stub = dfg_da_py::stub_info()?;
    stub.generate()?;
    Ok(())
}
