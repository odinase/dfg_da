if (CMAKE_VERSION VERSION_LESS 2.8.11)
  include_directories("${gtest_SOURCE_DIR}/include")
endif()


# Include tests here...
add_executable(test_dfg test_dfg.cpp)
add_executable(test_own_fg test_own_fg.cpp)

if (APPLE)
  set(GTEST_LIBS
    /usr/local/lib/libgtest.a
    /usr/local/lib/libgtest_main.a
  )
else()
  set(GTEST_LIBS
    gtest
    gtest_main
)
endif()


target_link_libraries(test_dfg
  gtsam
  hypothesis
  factor_graph
  glog::glog
  # lbp
  ${GTEST_LIBS}
)

add_test(TestDFG test_dfg)


target_link_libraries(test_own_fg
  gtsam
  # lbp
  factor_graph
  ${GTEST_LIBS}
)

add_test(Test_own_FG test_own_fg)