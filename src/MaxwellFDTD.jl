module MaxwellFDTD

import Adapt: adapt_storage, @adapt_structure, adapt
import CUDA: CUDA, CuArray, synchronize
import HDF5
import Interpolations: linear_interpolation, Flat
import KernelAbstractions: KernelAbstractions, @index, @kernel, get_backend
import ProgressMeter: @showprogress
import TimerOutputs: @timeit, reset_timer!, print_timer


export CPU, GPU, Grid1D, Grid2D, Grid3D, SoftSource, HardSource, TFSFSource, solve!, Model,
       Material, DebyeSusceptibility, DrudeSusceptibility, LorentzSusceptibility, Plasma,
       CPML, FieldMonitor


using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val
const EPS0 = VacuumElectricPermittivity.val
const MU0 = VacuumMagneticPermeability.val
const QE = ElementaryCharge.val
const ME = ElectronMass.val
const HBAR = ReducedPlanckConstant.val

CUDA.allowscalar(false)

include("backends.jl")
include("grids.jl")
include("pml.jl")
include("fields.jl")
include("materials.jl")
include("models.jl")
include("sources.jl")
include("monitors.jl")
include("outputs.jl")
include("util.jl")

end
