cmake_minimum_required(VERSION 3.10)
project(discrete_factor_graph)

set(CMAKE_CXX_STANDARD 17)

find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(dcsam REQUIRED)
find_package(idbt REQUIRED)


include_directories(
    ${GTSAM_INCLUDE_DIR}
    ${Eigen3_INCLUDE_DIR}
)

add_executable(${PROJECT_NAME}
    main.cpp
)

target_link_libraries(${PROJECT_NAME}
    gtsam
    gtsam_unstable
    dcsam
    idbt
    Eigen3::Eigen
)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY})
