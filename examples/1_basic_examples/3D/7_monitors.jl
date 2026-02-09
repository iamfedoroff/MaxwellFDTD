# ******************************************************************************************
# Demonstration of monitors.
# Here we create three monitors at points (0,0,-20um), (0,0,0), and (0,0,20um).
# These monitors record the fields passing through this points in all time moments.
# In particular, with this method we can monitor the polarization of radiation in a given
# point by plotting the fields in coordinates t, Ex, Ey.
# ******************************************************************************************
using MaxwellFDTD
using MaxwellPlots

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val

grid = Grid(
    xmin=-25e-6, xmax=25e-6, Nx=201,
    ymin=-25e-6, ymax=25e-6, Ny=201,
    zmin=-25e-6, zmax=25e-6, Nz=201,
)

function source_geometry(x, y, z)
    return z == -20e-6   # source plane at z=-20um
end

function source_waveform(x, y, z, t)
    lam0 = 2e-6   # (m) central wavelength
    w0 = 2*pi * C0 / lam0   # central frequency
    tau0 = 20e-15   # (s) pulse duration
    dt0 = 5 * tau0   # (s) time shift needed, since the simulation starts at time t=0
    a0 = 10e-6   # (m) beam radius

    # A = 1   # plane wave
    A = exp(-(x^2 + y^2) / a0^2)    # Gaussian beam

    return A * exp(-(t - dt0)^2 /tau0^2) * cos(w0 * (t - dt0))
end

source = SoftSource(geometry=source_geometry, waveform=source_waveform, component=:Ex)

function material_geometry(x, y, z)
    return z >= 0   # material occupies half-space z>0
end

# plot_geometry(grid.x, grid.y, grid.z, material_geometry)   # visualize material geometry

glass = Material(geometry=material_geometry, eps=1.5^2)

model = Model(grid, source, tmax=400e-15, pml=5e-6, material=glass)

monitor1_geometry(x,y,z) = x == 0 && y == 0 && abs(z - -20e-6) <= grid.dz/2
monitor2_geometry(x,y,z) = x == 0 && y == 0 && z == 0
monitor3_geometry(x,y,z) = x == 0 && y == 0 && abs(z - 20e-6) <= grid.dz/2
monitors = (
    FieldMonitor(monitor1_geometry),
    FieldMonitor(monitor2_geometry),
    FieldMonitor(monitor3_geometry),
)

# backend = CPU()   # simulate on CPU
backend = GPU()   # simulate on GPU
solve!(model; backend, nframes=50, monitors)

plot_monitors("out.hdf", :Ex, (1,2,3))   # electric field over time
# plot_monitors_spectrum("out.hdf", :Ex, (1,2,3))   # electric field spectrum
# plot_monitors_polarization("out.hdf", :Ex, :Ey, (1,2,3))   # polarization
