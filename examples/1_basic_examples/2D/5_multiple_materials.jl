# ******************************************************************************************
# Interaction of a Gaussian pulse with multiple dielectric materials.
# ******************************************************************************************
using MaxwellFDTD
using MaxwellPlots

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val

grid = Grid(
    xmin=-50e-6, xmax=50e-6, Nx=501,
    zmin=-50e-6, zmax=50e-6, Nz=1001,
)

function source_geometry(x, z)
    return z == -40e-6   # source line at z=-40um
end

function source_waveform(x, z, t)
    lam0 = 2e-6   # (m) central wavelength
    w0 = 2*pi * C0 / lam0   # central frequency
    tau0 = 20e-15   # (s) 1/e pulse duration
    dt0 = 5 * tau0   # (s) time shift needed, since the simulation starts at time t=0
    a0 = 10e-6   # (m) beam radius

    # A = 1   # plane wave
    A = exp(-x^2 / a0^2)   # Gaussian beam

    return A * exp(-(t - dt0)^2 /tau0^2) * cos(w0 * (t - dt0))
end

source = SoftSource(geometry=source_geometry, waveform=source_waveform, component=:Ex)

material_geometry_1(x, z) = z >= 0 && z <= 25e-6
m1 = Material(geometry=material_geometry_1, eps=1.5^2)

material_geometry_2(x, z) = z >= 5e-6 && z <= 20e-6
m2 = Material(geometry=material_geometry_2, eps=2^2)

material_geometry_3(x, z) = z >= 10e-6 && z <= 15e-6
m3 = Material(geometry=material_geometry_3, eps=2.5^2)

model = Model(grid, source, tmax=1200e-15, pml=5e-6, material=(m1,m2,m3))

# plot_geometry(grid.x, grid.z, model.geometry)  # visualize material geometry

# mybackend = CPU()   # simulate on CPU
mybackend = GPU()   # simulate on GPU
solve!(model, nframes=100, backend=mybackend)

inspect("out.hdf", :Ex, colorrange=(-0.1,0.1))
