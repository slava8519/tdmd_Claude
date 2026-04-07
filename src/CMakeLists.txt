# src/CMakeLists.txt
# Each module is built as an OBJECT library; the final executable and tests link against them.

# ---- subdirectories (each adds its own library target) ----
add_subdirectory(core)
add_subdirectory(io)
add_subdirectory(domain)
add_subdirectory(neighbors)
add_subdirectory(potentials)
add_subdirectory(integrator)
# add_subdirectory(scheduler)   # M3
# add_subdirectory(comm)        # M5
# add_subdirectory(telemetry)   # M0+
add_subdirectory(drivers)
