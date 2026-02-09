using MaxwellFDTD
using MaxwellPlots

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val
const EPS0 = VacuumElectricPermittivity.val


# ******************************************************************************************
# Grid
# ******************************************************************************************
grid = Grid(
    xmin=-40e-6, xmax=40e-6, Nx=401,
    zmin=-5e-6, zmax=105e-6, Nz=1101,
)


# ******************************************************************************************
# Source
# ******************************************************************************************
lam0 = 1.03e-6   # (m) central wavelength
tau0 = 20e-15   # (s) pulse duration
dt0 = 5 * tau0   # (s) time shift needed, since the simulation starts at time t=0
a0 = 10e-6   # (m) beam radius

w0 = 2*pi * C0 / lam0   # central frequency

n0 = 1.45   # refractive index at 1.03um [Malitson, JOSA, 55, 1205 (1965)]
I0 = 5e13*1e4   # (W/m^2) initial intensity
E0 = sqrt(I0 / (n0 * EPS0 * C0 / 2))   # initial field amplitude

params = (w0, tau0, dt0, a0, E0)

function source_waveform(x, z, t, p)
    w0, tau0, dt0, a0, E0 = p
    A = exp(-x^2 / a0^2)   # Gaussian beam
    return E0 * A * exp(-(t - dt0)^2 / tau0^2) * cos(w0 * (t - dt0))
end

function source_geometry(x, z)
    return z == 0
end

source = SoftSource(
    geometry=source_geometry, waveform=source_waveform, p=params, component=:Ex,
)


# ******************************************************************************************
# Material
# ******************************************************************************************
# Kerr -------------------------------------------------------------------------------------
n2 = 3.54e-20   # (m^2/W) nonlinear index [Couairon, PRB, 71, 125435 (2005)]
chi3 = 4/3 * n0^2 * EPS0 * C0 * n2   # cubic susceptibility


# Plasma -----------------------------------------------------------------------------------
include("IonizationRate.jl")
ionrate_keldysh = IonizationRate("SiO2_keldysh_1.03um.tf")

# ionrate_mpi(I) = 1e-130*I^8   # MPI ionization rate
ionrate_mpi(I) = 10^(8 * log10(I) + log10(1e-130))   # MPI ionization rate (log scale)

# ionrate = ionrate_keldysh
ionrate = ionrate_mpi

rho0 = 2.1e28   # (1/m^3) neutrals density
nuc = 1 / 2.33e-14   # (1/s) collision frequency [Sudrie, PRL, 89, 186601 (2002)]
frequency = w0   # (1/s) frequency at which the plasma response is calculated
Uiev = 9.0   # (eV) ionization potential (bandgap)
mr = 0.64   # (ME) effective mass [Couairon, PRB, 71, 125435 (2005)]
plasma = Plasma(; ionrate, rho0, nuc, frequency, Uiev, mr)


#-------------------------------------------------------------------------------------------
function material_geometry(x, z)
    return true   # material occupies whole space
    # return z >= 50e-6   # material occupies half-space z>50um
    # return z >= 50e-6 && z <= 70e-6   # material slab from z=50 to z=70um
end

# plot_geometry(grid.z, material_geometry)   # visualize material geometry

SiO2 = Material(geometry=material_geometry, eps=n0^2, chi3=chi3, plasma=plasma)


# ******************************************************************************************
# Model & Solve
# ******************************************************************************************
model = Model(grid, source, tmax=700e-15, pml=5e-6, material=SiO2)

solve!(model, nframes=100, backend=GPU())


# ******************************************************************************************
# Visualization
# ******************************************************************************************
inspect("out.hdf", :Ex)
# inspect("out.hdf", :rho)
# plot2D("out.hdf", :Sa)
# plot2D("out.hdf", :rho_end)
# plot2D("out.hdf", :JE)
