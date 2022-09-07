if (CMAKE_VERSION VERSION_LESS 2.8.11)
  include_directories("${gtest_SOURCE_DIR}/include")
endif()


# Include tests here...
add_executable(test_dfg test_dfg.cpp)

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
  lbp
  ${GTEST_LIBS}
)

add_test(TestDFG test_dfg)