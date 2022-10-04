cmake_minimum_required(VERSION 3.10)
project(discrete_factor_graph)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Wall -Wpedantic -Wextra)


# Add option to enable testing
option(ENABLE_TESTS "Enable tests" OFF)

find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED)
# find_package(dcsam REQUIRED)
# find_package(idbt REQUIRED)

find_package (glog REQUIRED)
find_package(matioCpp)

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



add_library(factor_graph
    src/factor_graph.cpp
    src/nodes.cpp
)


target_link_libraries(factor_graph
    gtsam
    Eigen3::Eigen
    hypothesis
)

# add_library(lbp
#     src/lbp.cpp
# )

# target_link_libraries(lbp
#     gtsam
#     Eigen3::Eigen 
# )


add_executable(${PROJECT_NAME}
    main.cpp
)

target_link_libraries(${PROJECT_NAME}
    gtsam
    # lbp
    factor_graph
    # gtsam_unstable
    # dcsam
    # idbt
    Eigen3::Eigen
    glog::glog
)


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
    # lbp
    factor_graph
    # gtsam_unstable
    # dcsam
    # idbt
    Eigen3::Eigen
    glog::glog
)


# Include unit tests directory to the project.
if (${ENABLE_TESTS})
message(STATUS "Testing enabled. Building tests.")
enable_testing()
add_subdirectory(tests)
endif()


set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(test_fg_lbp PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

if (matioCpp_FOUND)
set_target_properties(at612_reader PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
endif (matioCpp_FOUND)