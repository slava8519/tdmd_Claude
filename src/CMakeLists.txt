# src/CMakeLists.txt
# Each module is built as an OBJECT library; the final executable and tests link against them.

# ---- tdmd::core ----
add_library(tdmd_core INTERFACE)
target_include_directories(tdmd_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tdmd_core INTERFACE tdmd_flags)

# ---- subdirectories (each adds its own library target) ----
add_subdirectory(core)
# add_subdirectory(io)          # M1
# add_subdirectory(domain)      # M1
# add_subdirectory(neighbors)   # M1
# add_subdirectory(potentials)  # M1
# add_subdirectory(integrator)  # M1
# add_subdirectory(scheduler)   # M3
# add_subdirectory(comm)        # M5
# add_subdirectory(telemetry)   # M0+
add_subdirectory(drivers)
