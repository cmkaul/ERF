# Additional testing options
set(ERF_TEST_NRANKS 2 CACHE STRING  "Number of MPI ranks to use for each test")
set(ERF_TEST_FCOMPARE_RTOL "2.0e-10" CACHE STRING "fcompare relative tolerance")
set(ERF_TEST_FCOMPARE_ATOL "2.0e-10" CACHE STRING "fcompare absolute tolerance")
set(ERF_TEST_GOLD_FILES_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ERFGoldFiles" CACHE PATH "Path to ERF gold files")
option(ERF_TEST_ENABLE_EXTRA_SDM_TESTS "Enable additional SDM tests that require machine-specific gold files" OFF)

message(STATUS "ERF testing configuration summary:")
message(STATUS "   Number of ranks               = ${ERF_TEST_NRANKS}")
message(STATUS "   fcompare executable           = ${FCOMPARE_EXE}")
message(STATUS "   comparison relative tolerance = ${ERF_TEST_FCOMPARE_RTOL}")
message(STATUS "   comparison absolute tolerance = ${ERF_TEST_FCOMPARE_ATOL}")
message(STATUS "   Gold file directory           = ${ERF_TEST_GOLD_FILES_DIRECTORY}")
message(STATUS "   Extra SDM tests enabled       = ${ERF_TEST_ENABLE_EXTRA_SDM_TESTS}")

include(${CMAKE_CURRENT_SOURCE_DIR}/CTestList.cmake)

# ---------------------------------------------------------------------------
# Lightweight unit tests (host-only executables)
# ---------------------------------------------------------------------------
add_executable(erf_unit_realbdy_interp Unit/ERF_RealBdyInterpUnit.cpp)
target_include_directories(erf_unit_realbdy_interp PRIVATE
  ${CMAKE_SOURCE_DIR}/Source
  ${CMAKE_SOURCE_DIR}/Source/Utils
)
target_link_libraries(erf_unit_realbdy_interp PRIVATE amrex_3d)
add_test(NAME ERF_RealBdyInterpUnit COMMAND $<TARGET_FILE:erf_unit_realbdy_interp>)
set_tests_properties(ERF_RealBdyInterpUnit PROPERTIES LABELS "unit")
