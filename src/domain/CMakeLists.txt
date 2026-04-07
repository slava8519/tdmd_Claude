# domain module — cell list, zone partitioning
add_library(tdmd_domain STATIC cell_list.cpp)
target_include_directories(tdmd_domain PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(tdmd_domain PUBLIC tdmd::core)
add_library(tdmd::domain ALIAS tdmd_domain)
