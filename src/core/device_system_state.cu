// SPDX-License-Identifier: Apache-2.0
// device_system_state.cu — implementation of DeviceSystemState.
#include "device_system_state.cuh"

namespace tdmd {

void DeviceSystemState::resize(i64 n, i32 ntypes) {
  natoms = n;
  auto un = static_cast<std::size_t>(n);
  positions.resize(un);
  velocities.resize(un);
  forces.resize(un);
  types.resize(un);
  ids.resize(un);
  if (ntypes > 0) {
    masses.resize(static_cast<std::size_t>(ntypes));
  }
}

void DeviceSystemState::upload(const SystemState& host) {
  natoms = host.natoms;
  box = host.box;
  auto n = static_cast<std::size_t>(host.natoms);

  // Resize if needed.
  if (positions.size() != n) {
    resize(host.natoms,
           static_cast<i32>(host.masses.size()));
  }

  positions.copy_from_host(host.positions.data(), n);
  velocities.copy_from_host(host.velocities.data(), n);
  forces.copy_from_host(host.forces.data(), n);
  types.copy_from_host(host.types.data(), n);
  ids.copy_from_host(host.ids.data(), n);
  if (!host.masses.empty()) {
    masses.copy_from_host(host.masses.data(), host.masses.size());
  }
}

void DeviceSystemState::download(SystemState& host) const {
  host.natoms = natoms;
  host.box = box;
  auto n = static_cast<std::size_t>(natoms);

  host.resize(natoms);
  positions.copy_to_host(host.positions.data(), n);
  velocities.copy_to_host(host.velocities.data(), n);
  forces.copy_to_host(host.forces.data(), n);
  types.copy_to_host(host.types.data(), n);
  ids.copy_to_host(host.ids.data(), n);
  if (masses.size() > 0) {
    host.masses.resize(masses.size());
    masses.copy_to_host(host.masses.data(), masses.size());
  }
}

void DeviceSystemState::zero_forces() { forces.zero(); }

}  // namespace tdmd
