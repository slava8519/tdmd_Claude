# tests/CMakeLists.txt
# GoogleTest via FetchContent. No external install needed.

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(CTest)
include(GoogleTest)

# ---- Smoke test (legacy, still useful) ----
add_executable(test_smoke unit/test_smoke.cpp)
target_link_libraries(test_smoke PRIVATE tdmd::core)
add_test(NAME test_smoke COMMAND test_smoke)

# ---- Unit tests with GoogleTest ----
set(TDMD_TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data")

add_executable(tdmd_unit_tests
  unit/test_math.cpp
  unit/test_lammps_data_reader.cpp
  unit/test_cell_list.cpp
  unit/test_neighbor_list.cpp
  unit/test_morse.cpp
  unit/test_velocity_verlet.cpp
  unit/test_eam.cpp
  unit/test_lammps_ab.cpp
  unit/test_nve_drift.cpp
)
target_link_libraries(tdmd_unit_tests PRIVATE tdmd::core tdmd::io tdmd::domain tdmd::neighbors tdmd::potentials tdmd::integrator GTest::gtest_main)
target_compile_definitions(tdmd_unit_tests PRIVATE
  TDMD_TEST_DATA_DIR="${TDMD_TEST_DATA_DIR}"
)
gtest_discover_tests(tdmd_unit_tests)
