import CUDA
using MaxwellFDTD
using Test

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val

fname = joinpath(@__DIR__, "out.hdf")

include("1d_test.jl")
include("2d_test.jl")
include("3d_test.jl")

rm(fname)
