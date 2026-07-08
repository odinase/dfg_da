cmake_minimum_required(VERSION 3.16)
project(dfg_da_c CXX)

# Export compile_commands.json for clangd / clang-tidy linting
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# The C-API wrapper compiled together with the (GTSAM-free) dfg_da sources it
# wraps -- mirrors the direct g++ build of the py_dfg_da extension, so this needs
# only Eigen + the stdlib (no GTSAM, no pybind11).
set(DFG_DA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(PYEHM_ROOT ${DFG_DA_ROOT}/third_party/pyehm)

# Use pyehm's vendored Eigen (>= 3.4; the EHM core uses Eigen::all) for everything
# here, so a single Eigen version/ABI is shared across the LBP sources, the EHM2
# C-API, and pyehm_core -- and no system Eigen (libeigen3-dev) is required.
set(EIGEN_INCLUDE ${PYEHM_ROOT}/include)
if(NOT EXISTS ${EIGEN_INCLUDE}/Eigen/Dense)
    message(FATAL_ERROR
        "Vendored Eigen not found at ${EIGEN_INCLUDE}. "
        "Run: git submodule update --init third_party/pyehm")
endif()

# --- pyehm C++ core (EHM/EHM2), GTSAM-free, Eigen + stdlib only ---
# The pure C++ core of the pyehm fork, without its pybind11 glue
# (src/module.cpp, src/Docstrings.cpp). Mirrors pyehm's own setup.py, which
# globs src/{core,net,utils}/*.cpp with include dirs src/ + include/.
file(GLOB PYEHM_CORE_SOURCES CONFIGURE_DEPENDS
    ${PYEHM_ROOT}/src/core/*.cpp
    ${PYEHM_ROOT}/src/net/*.cpp
    ${PYEHM_ROOT}/src/utils/*.cpp
)

if(NOT PYEHM_CORE_SOURCES)
    message(FATAL_ERROR
        "pyehm sources not found under ${PYEHM_ROOT}/src. "
        "Run: git submodule update --init third_party/pyehm")
endif()

add_library(pyehm_core STATIC ${PYEHM_CORE_SOURCES})
target_include_directories(pyehm_core PUBLIC ${PYEHM_ROOT}/src ${EIGEN_INCLUDE})
set_target_properties(pyehm_core PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Static: Rust links it straight into each dependent binary, so there is no
# runtime .so to locate (no rpath/LD_LIBRARY_PATH dance). PIC below keeps it
# usable inside the pyo3 cdylib too.
add_library(dfg_da_c STATIC
    src/lbp_c.cpp
    src/ehm2.cpp
    ${DFG_DA_ROOT}/src/hypothesis.cpp
    ${DFG_DA_ROOT}/src/lbp.cpp
)

target_include_directories(dfg_da_c PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_include_directories(dfg_da_c PRIVATE
    ${DFG_DA_ROOT}/include
    ${PYEHM_ROOT}/src
    ${EIGEN_INCLUDE}
)

target_link_libraries(dfg_da_c PRIVATE pyehm_core)

set_target_properties(dfg_da_c PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    # Place libdfg_da_c.a directly in the build dir for easy discovery by build.rs.
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
