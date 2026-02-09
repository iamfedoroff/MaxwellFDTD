# ******************************************************************************************
# Interaction with a simple dielectric material.
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

function material_geometry(x, z)
    # return z >= -20e-6   # material occupies half-space z>0
    # return z >= 0 && z <= 20e-6   # material slab from z=0 to z=20um
    # return sqrt(x^2 + z^2) <= 10e-6   # material is inside circle of radius 10um

    # Inclined interface passing through points (x,z)=(0,0) and (-50um,50um):
    x1, z1 = 0, 0
    x2, z2 = -50e-6, 50e-6
    dzdx = (z2 - z1) / (x2 - x1)
    return (z - z1) >= dzdx * (x - x1)
end

# plot_geometry(grid.x, grid.z, material_geometry)   # visualize material geometry

glass = Material(geometry=material_geometry, eps=1.5^2)

model = Model(grid, source, tmax=600e-15, pml=5e-6, material=glass)

# mybackend = CPU()   # simulate on CPU
mybackend = GPU()   # simulate on GPU
solve!(model, nframes=100, backend=mybackend)

inspect("out.hdf", :Ex)
