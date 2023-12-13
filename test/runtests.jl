import CUDA
import HDF5
using MaxwellFDTD
using Test

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val
const EPS0 = VacuumElectricPermittivity.val

fname = joinpath(@__DIR__, "out.hdf")

include("1d_test.jl")
include("2d_test.jl")
include("3d_test.jl")
include("1d_test_losses.jl")
include("2d_test_losses.jl")
include("3d_test_losses.jl")

rm(fname)
