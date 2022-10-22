cmake_minimum_required(VERSION 3.10)
project(
    dfg_da
    VERSION 0.0.1
)

# make cache variables for install destinations
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# TODO(odin): Is this necessary?
set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR})

add_compile_options(-Wall -Wpedantic -Wextra)


# Add option to enable testing
option(ENABLE_TESTS "Enable tests" OFF)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)


find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(matioCpp)
find_package (glog)


if (glog_FOUND)
add_definitions(-DGLOG_AVAILABLE)
message("Building with glog")
else()
message("Building without glog")
endif()

if (matioCpp_FOUND)
# add_definitions(-DGLOG_AVAILABLE)
message("Building with matioCpp")
else()
message("Building without matioCpp")
endif()


add_library(hypothesis SHARED
    src/hypothesis.cpp
)

target_include_directories(hypothesis PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(hypothesis PUBLIC
    gtsam
    Eigen3::Eigen 
)

add_library(${PROJECT_NAME} SHARED
    src/factor_graph.cpp
    src/nodes.cpp
    src/lbp
)


target_link_libraries(${PROJECT_NAME}
    gtsam
    Eigen3::Eigen
    hypothesis
)

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)


if(NOT ${BUILD_PYTHON_BINDINGS})

add_executable(discrete_factor_graph
    main.cpp
)

target_link_libraries(discrete_factor_graph
    gtsam
    ${PROJECT_NAME}
    Eigen3::Eigen
)

if(glog_FOUND)
target_link_libraries(discrete_factor_graph
  glog::glog
)
endif()


if (matioCpp_FOUND)
add_executable(at612_reader
    src/at612_reader.cpp
)

target_link_libraries(at612_reader
    gtsam
    factor_graph
    Eigen3::Eigen
    matioCpp::matioCpp
)
endif (matioCpp_FOUND)

add_executable(test_fg_lbp
    fg_lbp.cpp
)

target_link_libraries(test_fg_lbp
    gtsam
    hypothesis
    ${PROJECT_NAME}
    Eigen3::Eigen
)


if(glog_FOUND)
target_link_libraries(test_fg_lbp
  glog::glog
)
endif()


set_target_properties(discrete_factor_graph PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(test_fg_lbp PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

endif(NOT ${BUILD_PYTHON_BINDINGS})

# Include unit tests directory to the project.
if (${ENABLE_TESTS})
message(STATUS "Testing enabled. Building tests.")
enable_testing()
add_subdirectory(tests)
endif()

if(${BUILD_PYTHON_BINDINGS})
message(STATUS "Building Python bindings.")
add_subdirectory(pybindings)
endif(${BUILD_PYTHON_BINDINGS})

if (matioCpp_FOUND)
set_target_properties(at612_reader PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
endif (matioCpp_FOUND)

if (NOT ${BUILD_PYTHON_BINDINGS})
# locations are provided by GNUInstallDirs
install(TARGETS ${PROJECT_NAME} hypothesis
        EXPORT ${PROJECT_NAME}_Targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file("${PROJECT_NAME}ConfigVersion.cmake"
                                    VERSION ${PROJECT_VERSION}
                                    COMPATIBILITY ExactVersion)

configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION
    ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

install(EXPORT ${PROJECT_NAME}_Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

install(FILES   "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME} DESTINATION include)
endif(NOT ${BUILD_PYTHON_BINDINGS})
