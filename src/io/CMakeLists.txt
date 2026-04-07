# io module — LAMMPS file readers/writers
add_library(tdmd_io STATIC lammps_data_reader.cpp)
target_include_directories(tdmd_io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(tdmd_io PUBLIC tdmd::core)
add_library(tdmd::io ALIAS tdmd_io)
