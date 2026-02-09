# ******************************************************************************************
# Interaction of a Gaussian pulse with multiple dielectric materials.
# ******************************************************************************************
using MaxwellFDTD
using MaxwellPlots

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val

grid = Grid(
    xmin=-25e-6, xmax=25e-6, Nx=201,
    ymin=-25e-6, ymax=25e-6, Ny=201,
    zmin=-25e-6, zmax=25e-6, Nz=401,
)

function source_geometry(x, y, z)
    return z == -20e-6   # source plane at z=-20um
end

function source_waveform(x, y, z, t)
    lam0 = 2e-6   # (m) central wavelength
    w0 = 2*pi * C0 / lam0   # central frequency
    tau0 = 30e-15   # (s) FWHM pulse duration
    a0 = 10e-6   # (m) beam radius

    # A = 1   # plane wave
    A = exp(-(x^2 + y^2) / a0^2)    # Gaussian beam

    # sin2 pulse:
    T = t > 2*tau0 ? 0 : sin(pi * (t - 2*tau0) / (2*tau0))^2 * cos(w0 * (t - 2*tau0))

    return A * T
end

source = SoftSource(geometry=source_geometry, waveform=source_waveform, component=:Ex)

material_geometry_1(x, y, z) = z >= 0 && z <= 25e-6
m1 = Material(geometry=material_geometry_1, eps=1.5^2)

material_geometry_2(x, y, z) = z >= 5e-6 && z <= 20e-6
m2 = Material(geometry=material_geometry_2, eps=2^2)

material_geometry_3(x, y, z) = z >= 10e-6 && z <= 15e-6
m3 = Material(geometry=material_geometry_3, eps=2.5^2)

model = Model(grid, source, tmax=400e-15, pml=5e-6, material=(m1,m2,m3))

# plot_geometry(grid.x, grid.y, grid.z, model.geometry)   # visualize material geometry

# mybackend = CPU()   # simulate on CPU
mybackend = GPU()   # simulate on GPU
solve!(model, nframes=50, backend=mybackend)

inspect("out.hdf", :Ex, colorrange=(-0.1,0.1))
