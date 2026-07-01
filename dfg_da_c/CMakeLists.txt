cmake_minimum_required(VERSION 3.16)
project(dfg_da_c CXX)

find_package(Eigen3 REQUIRED)

# The C-API wrapper compiled together with the (GTSAM-free) dfg_da sources it
# wraps -- mirrors the direct g++ build of the py_dfg_da extension, so this needs
# only Eigen + the stdlib (no GTSAM, no pybind11).
set(DFG_DA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Static: Rust links it straight into each dependent binary, so there is no
# runtime .so to locate (no rpath/LD_LIBRARY_PATH dance). PIC below keeps it
# usable inside the pyo3 cdylib too.
add_library(dfg_da_c STATIC
    src/lbp_c.cpp
    ${DFG_DA_ROOT}/src/hypothesis.cpp
    ${DFG_DA_ROOT}/src/lbp.cpp
)

target_include_directories(dfg_da_c PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_include_directories(dfg_da_c PRIVATE
    ${DFG_DA_ROOT}/include
)

target_link_libraries(dfg_da_c PRIVATE Eigen3::Eigen)

set_target_properties(dfg_da_c PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    # Place libdfg_da_c.a directly in the build dir for easy discovery by build.rs.
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
