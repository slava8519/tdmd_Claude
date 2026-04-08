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
  unit/test_zone_state_machine.cpp
  unit/test_zone_partition.cpp
  unit/test_sequential_scheduler.cpp
)
target_link_libraries(tdmd_unit_tests PRIVATE tdmd::core tdmd::io tdmd::domain tdmd::neighbors tdmd::potentials tdmd::integrator tdmd::scheduler GTest::gtest_main)
target_compile_definitions(tdmd_unit_tests PRIVATE
  TDMD_TEST_DATA_DIR="${TDMD_TEST_DATA_DIR}"
)
gtest_discover_tests(tdmd_unit_tests)

# ---- CUDA unit tests (M2+) ----
if(TDMD_ENABLE_CUDA)
  add_executable(tdmd_cuda_tests
    unit/test_device_buffer.cu
    unit/test_device_cell_list.cu
    unit/test_device_neighbor_list.cu
    unit/test_device_morse.cu
    unit/test_device_eam.cu
    unit/test_device_lammps_ab.cu
    unit/test_device_nve_drift.cu
    unit/test_pipeline_scheduler.cu
    unit/test_nvt_thermostat.cu
  )
  target_link_libraries(tdmd_cuda_tests PRIVATE
    tdmd::core tdmd::io tdmd::domain tdmd::neighbors tdmd::potentials tdmd::integrator tdmd::scheduler
    GTest::gtest_main
  )
  target_compile_definitions(tdmd_cuda_tests PRIVATE
    TDMD_TEST_DATA_DIR="${TDMD_TEST_DATA_DIR}"
  )
  set_target_properties(tdmd_cuda_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )
  gtest_discover_tests(tdmd_cuda_tests)

  # ---- MPI + CUDA tests (M5+) ----
  if(TDMD_ENABLE_MPI)
    add_executable(tdmd_mpi_tests
      unit/test_distributed_pipeline.cu
    )
    target_link_libraries(tdmd_mpi_tests PRIVATE
      tdmd::core tdmd::io tdmd::domain tdmd::neighbors tdmd::potentials tdmd::integrator tdmd::scheduler tdmd::comm
      GTest::gtest
      MPI::MPI_CXX
    )
    target_compile_definitions(tdmd_mpi_tests PRIVATE
      TDMD_TEST_DATA_DIR="${TDMD_TEST_DATA_DIR}"
    )
    set_target_properties(tdmd_mpi_tests PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
    )
    # MPI tests must be run with mpirun, not gtest_discover_tests.
    add_test(NAME DistributedPipeline.DeterministicMatchesSingleRank
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:tdmd_mpi_tests> --gtest_filter=DistributedPipeline.DeterministicMatchesSingleRank)
    add_test(NAME DistributedPipeline.PipelineNVEConservation
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:tdmd_mpi_tests> --gtest_filter=DistributedPipeline.PipelineNVEConservation)
    add_test(NAME DistributedPipeline.ZoneTimeStepsAdvance
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:tdmd_mpi_tests> --gtest_filter=DistributedPipeline.ZoneTimeStepsAdvance)

    # ---- M6 Hybrid 2D tests (4 ranks: P_time=2, P_space=2) ----
    add_executable(tdmd_m6_tests
      unit/test_hybrid_pipeline.cu
    )
    target_link_libraries(tdmd_m6_tests PRIVATE
      tdmd::core tdmd::io tdmd::domain tdmd::neighbors tdmd::potentials tdmd::integrator tdmd::scheduler tdmd::comm
      GTest::gtest
      MPI::MPI_CXX
    )
    target_compile_definitions(tdmd_m6_tests PRIVATE
      TDMD_TEST_DATA_DIR="${TDMD_TEST_DATA_DIR}"
    )
    set_target_properties(tdmd_m6_tests PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME HybridPipeline.DeterministicMatchesM5
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:tdmd_m6_tests> --gtest_filter=HybridPipeline.DeterministicMatchesM5)
    add_test(NAME HybridPipeline.PipelineNVEConservation
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:tdmd_m6_tests> --gtest_filter=HybridPipeline.PipelineNVEConservation)
    add_test(NAME HybridPipeline.ZoneTimeStepsAdvance
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:tdmd_m6_tests> --gtest_filter=HybridPipeline.ZoneTimeStepsAdvance)
  endif()
endif()
