# ******************************************************************************************
# Excitation of Ex and Hy field components at the same time to obtain the source emitting
# radiation only in one direction (a small part emitted in the opposite direction is due to
# the numerical dispersion).
# ******************************************************************************************
using MaxwellFDTD
using MaxwellPlots

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val
const EPS0 = VacuumElectricPermittivity.val
const MU0 = VacuumMagneticPermeability.val

grid = Grid(zmin=-50e-6, zmax=50e-6, Nz=2001)

function source_geometry(z)
    return z == 0   # point source at z=0
    # return z == -20e-6 || z == 20e-6  # two point sources at z=-20um and z=20um
    # return abs(z) <= 10e-6   # distributed source in the range from z=-10um to z=10um
end

function source_waveform_Ex(z, t)
    lam0 = 2e-6   # (m) central wavelength
    w0 = 2*pi * C0 / lam0   # central frequency
    tau0 = 20e-15   # (s) pulse duration
    dt0 = 5 * tau0   # (s) time shift needed, since the simulation starts at time t=0
    return exp(-(t - dt0)^2 /tau0^2) * cos(w0 * (t - dt0)) / 2
end

function source_waveform_Hy(z, t)
    lam0 = 2e-6   # (m) central wavelength
    w0 = 2*pi * C0 / lam0   # central frequency
    tau0 = 20e-15   # (s) pulse duration
    dt0 = 5 * tau0   # (s) time shift needed, since the simulation starts at time t=0
    return sqrt(EPS0/MU0) * exp(-(t - dt0)^2 /tau0^2) * cos(w0 * (t - dt0)) / 2
end

source = (
    SoftSource(geometry=source_geometry, waveform=source_waveform_Ex, component=:Ex),
    SoftSource(geometry=source_geometry, waveform=source_waveform_Hy, component=:Hy),
)

model = Model(grid, source, tmax=500e-15, pml=5e-6)

solve!(model, nframes=100)

inspect("out.hdf", :Ex)
