// SPDX-License-Identifier: Apache-2.0
// mpi_types.hpp — compile-time MPI_Datatype selection for TDMD scalar types.
//
// Replaces runtime `sizeof(real) == 8 ? MPI_DOUBLE : MPI_FLOAT` patterns.
// See ADR 0007 §MPI types.
#pragma once

#include <mpi.h>

namespace tdmd {

/// @brief Returns MPI_Datatype matching the C++ type T.
template<class T>
MPI_Datatype mpi_type();

template<> inline MPI_Datatype mpi_type<float>()  { return MPI_FLOAT; }
template<> inline MPI_Datatype mpi_type<double>() { return MPI_DOUBLE; }
template<> inline MPI_Datatype mpi_type<int>()    { return MPI_INT; }

}  // namespace tdmd
