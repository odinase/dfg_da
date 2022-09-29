cmake_minimum_required(VERSION 3.10)
project(
    dfg_da
    VERSION 0.0.1
)

# make cache variables for install destinations
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_compile_options(-Wall -Wpedantic -Wextra)


# Add option to enable testing
option(ENABLE_TESTS "Enable tests" OFF)

find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED)
# find_package(dcsam REQUIRED)
# find_package(idbt REQUIRED)

find_package (glog)


if (glog_FOUND)
add_definitions(-DGLOG_AVAILABLE)
message("Building with glog")
else()
message("Building without glog")
endif()

include_directories(
    ${GTSAM_INCLUDE_DIR}
    ${Eigen3_INCLUDE_DIR}
    include
)

add_library(hypothesis
    src/hypothesis.cpp
)

target_link_libraries(hypothesis
    gtsam
    Eigen3::Eigen 
)

add_library(${PROJECT_NAME} STATIC
    src/hypothesis.cpp
    src/factor_graph.cpp
    src/nodes.cpp
)


target_link_libraries(${PROJECT_NAME}
    gtsam
    Eigen3::Eigen
    # hypothesis
)

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    # ${GTSAM_INCLUDE_DIR}
    # ${Eigen3_INCLUDE_DIR}
    # include
)

# add_library(lbp
#     src/lbp.cpp
# )

# target_link_libraries(lbp
#     gtsam
#     Eigen3::Eigen 
# )


add_executable(discrete_factor_graph #${PROJECT_NAME}
    main.cpp
)

target_link_libraries(discrete_factor_graph #${PROJECT_NAME}
    gtsam
    # lbp
    ${PROJECT_NAME}
    # gtsam_unstable
    # dcsam
    # idbt
    Eigen3::Eigen
)


if(glog_FOUND)
target_link_libraries(discrete_factor_graph
  glog::glog
)
endif()

add_executable(test_fg_lbp
    fg_lbp.cpp
)

target_link_libraries(test_fg_lbp
    gtsam
    hypothesis
    # lbp
    ${PROJECT_NAME}
    # gtsam_unstable
    # dcsam
    # idbt
    Eigen3::Eigen
)


if(glog_FOUND)
target_link_libraries(test_fg_lbp
  glog::glog
)
endif()


# Include unit tests directory to the project.
if (${ENABLE_TESTS})
message(STATUS "Testing enabled. Building tests.")
enable_testing()
add_subdirectory(tests)
endif()


set_target_properties(discrete_factor_graph PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(test_fg_lbp PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})


# locations are provided by GNUInstallDirs
install(TARGETS ${PROJECT_NAME}
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
    # ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
)

install(EXPORT ${PROJECT_NAME}_Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
    # DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
)

install(FILES   "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        # DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/discrete_factor_graph DESTINATION include)