# cmake_minimum_required(VERSION 3.10)
# project(af_target_simulator)

set(PYTHON_MODULE_NAME ${PROJECT_NAME})


# set (CMAKE_CXX_STANDARD 17)

# set(CMAKE_CXX_FLAGS "-O2")

# find_package (Eigen3 3.3 REQUIRED NO_MODULE)

# include_directories(
#     include
#     ${CMAKE_CURRENT_SOURCE_DIR}
# )

# add_library(testing
# src/Test.cpp
# )

# target_link_libraries(testing PRIVATE
#     pybind11::module
#     pybind11::lto
#     pybind11::windows_extras
# )

# add_library(targets
# src/targets/Target.cpp
# )

# target_link_libraries(targets PRIVATE
#     Eigen3::Eigen
#     pybind11::module
#     pybind11::lto
#     pybind11::windows_extras
# )

# add_library(dynamics
# src/dynamics/DynamicModel.cpp
# )

# target_link_libraries(dynamics PRIVATE
#     Eigen3::Eigen
#     pybind11::module
#     pybind11::lto
#     pybind11::windows_extras
# )


# add_library(utils
# src/utils.cpp
# )

# target_link_libraries(utils PRIVATE
#     Eigen3::Eigen
#     pybind11::module
#     pybind11::lto
#     pybind11::windows_extras
# )

# add_library(wrappers
# wrappers/wrappers.cpp
# )

# target_link_libraries(wrappers PRIVATE
# testing
# targets
# dynamics
# utils
#     Eigen3::Eigen
#     pybind11::module
#     pybind11::lto
#     pybind11::windows_extras
# )

# add_executable(main
#     main.cpp
# )

# target_link_libraries(main
#     testing
# )

add_subdirectory(pybind11)
pybind11_add_module(py_${PYTHON_MODULE_NAME} module.cpp)

target_link_libraries(py_${PYTHON_MODULE_NAME} PRIVATE
    pybind11::module
    pybind11::lto
    pybind11::windows_extras
    # testing
    # targets
    # dynamics
    # utils
    # wrappers
    # Eigen3::Eigen
)