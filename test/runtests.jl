using MaxwellFDTD
using Test

import HDF5

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val

include("1d_test.jl")
