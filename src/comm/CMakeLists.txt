# comm module — MPI ring communicator for TD zone exchange
if(TDMD_ENABLE_MPI)
  add_library(tdmd_comm STATIC
    mpi_ring_comm.cpp
  )
  target_include_directories(tdmd_comm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
  target_link_libraries(tdmd_comm PUBLIC tdmd::core MPI::MPI_CXX)
  add_library(tdmd::comm ALIAS tdmd_comm)
endif()
