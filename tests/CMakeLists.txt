# tests/CMakeLists.txt
# Uses CMake's built-in CTest + a minimal in-tree test harness for M0.
# Switch to Catch2 or doctest at M1 when we actually have things to test.

include(CTest)

add_executable(test_smoke unit/test_smoke.cpp)
target_link_libraries(test_smoke PRIVATE tdmd::core)

add_test(NAME test_smoke COMMAND test_smoke)
