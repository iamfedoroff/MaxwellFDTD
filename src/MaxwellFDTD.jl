module MaxwellFDTD

import Adapt: adapt_storage, @adapt_structure, adapt
import CUDA: CUDA, CuArray, @cuda, launch_configuration, synchronize, threadIdx,
             blockIdx, blockDim, gridDim
import HDF5
import ProgressMeter: @showprogress
import TimerOutputs: @timeit, reset_timer!, print_timer

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val
const EPS0 = VacuumElectricPermittivity.val
const MU0 = VacuumMagneticPermeability.val

CUDA.allowscalar(false)

include("gpu.jl")
include("grids.jl")
include("pml.jl")
include("fields.jl")
include("sources.jl")
include("materials.jl")
include("models.jl")
include("outputs.jl")

end
