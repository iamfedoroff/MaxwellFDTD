# ******************************************************************************************
# Laser pulses with arbitrary polarization.
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

lam0 = 2e-6   # (m) central wavelength
tau0 = 20e-15   # (s) pulse duration
dt0 = 5 * tau0   # (s) time shift needed, since the simulation starts at time t=0

ellipticity = 0.2   # ellipticity of polarization (ratio of polarization ellipse semi-axes)
theta = deg2rad(45)   # polarization rotation relative to the x-axis

w0 = 2*pi * C0 / lam0   # central frequency

params = (w0, tau0, dt0, ellipticity, theta)

function waveform_x(x, y, z, t, p)
    w0, tau0, dt0, ellipticity, theta = p
    norm = sqrt(1 + ellipticity^2)
    A = exp(-(t - dt0)^2 /tau0^2) / norm
    Tx = A * cos(w0 * (t - dt0))
    Ty = ellipticity * A * sin(w0 * (t - dt0))
    return Tx * cos(theta) - Ty * sin(theta)
end

function waveform_y(x, y, z, t, p)
    w0, tau0, dt0, ellipticity, theta = p
    norm = sqrt(1 + ellipticity^2)
    A = exp(-(t - dt0)^2 /tau0^2) / norm
    Tx = A * cos(w0 * (t - dt0))
    Ty = ellipticity * A * sin(w0 * (t - dt0))
    return Tx * sin(theta) + Ty * cos(theta)
end

function source_geometry(x, y, z)
    return z == -20e-6   # source plane at z=-20um
end

source_x = SoftSource(
    geometry=source_geometry, waveform=waveform_x, p=params, component=:Ex,
)

source_y = SoftSource(
    geometry=source_geometry, waveform=waveform_y, p=params, component=:Ey,
)

source = (source_x, source_y)

model = Model(grid, source, tmax=350e-15, pml=5e-6)

monitors = (FieldMonitor((0,0,-20e-6)), FieldMonitor((0,0,0)), FieldMonitor((0,0,20e-6)))

# mybackend = CPU()   # simulate on CPU
mybackend = GPU()   # simulate on GPU
solve!(model; nframes=50, backend=mybackend, monitors)

# plot_monitors("out.hdf", :Ex, (1,2,3))   # electric field over time
# plot_monitors_spectrum("out.hdf", :Ex, (1,2,3))   # electric field spectrum
plot_monitors_polarization("out.hdf", :Ex, :Ey, (1,2,3))   # polarization
