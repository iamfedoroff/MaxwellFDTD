# ******************************************************************************************
# Simulation of the propagation of a polarized pulse through a q-plate.
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

ellipticity = 0.25   # ellipticity of polarization (ratio of polarization ellipse semi-axes)
theta = deg2rad(90)   # polarization rotation relative to the x-axis

# ellipticity, theta = 0, deg2rad(0)   # radial
# ellipticity, theta = 0, deg2rad(45)   # spiral
# ellipticity, theta = 0, deg2rad(90)   # azimuthal

# L. Marrucci et al., Phys. Rev. Lett., 96, 163905 (2006)
q = 1/2  # topological charge of the q-plate
alpha0 = 0   # initial angle of the q-plate optical axis

w0 = 2*pi * C0 / lam0   # central frequency

params = (w0, tau0, dt0, ellipticity, theta, q, alpha0)

function waveform_x(x, y, z, t, p)
    w0, tau0, dt0, ellipticity, theta, q, alpha0 = p

    # Before q-plate:
    norm = sqrt(1 + ellipticity^2)
    A = exp(-(t - dt0)^2 /tau0^2) / norm
    Ex0 = A * cos(w0 * (t - dt0))
    Ey0 = ellipticity * A * sin(w0 * (t - dt0))
    Ex = Ex0 * cos(theta) - Ey0 * sin(theta)
    Ey = Ex0 * sin(theta) + Ey0 * cos(theta)

    # After q-plate:
    alpha = q * atan(y,x) + alpha0
    return Ex * cos(2*alpha) + Ey * sin(2*alpha)
end

function waveform_y(x, y, z, t, p)
    w0, tau0, dt0, ellipticity, theta, q, alpha0 = p

    # Before q-plate:
    norm = sqrt(1 + ellipticity^2)
    A = exp(-(t - dt0)^2 /tau0^2) / norm
    Ex0 = A * cos(w0 * (t - dt0))
    Ey0 = ellipticity * A * sin(w0 * (t - dt0))
    Ex = Ex0 * cos(theta) - Ey0 * sin(theta)
    Ey = Ex0 * sin(theta) + Ey0 * cos(theta)

    # After q-plate:
    alpha = q * atan(y,x) + alpha0
    return Ex * sin(2*alpha) - Ey * cos(2*alpha)
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

material = Material(geometry=(x,y,z)->z>=0)

model = Model(grid, source, tmax=350e-15, pml=5e-6, material=material)


# radially distributed monitors:
rvp = [5e-6, 10e-6, 15e-6]
thetavp = [deg2rad((i-1)*360/12) for i=1:12]
mymonitors = [FieldMonitor((r*cos(th), r*sin(th), 0)) for r in rvp for th in thetavp]


# mybackend = CPU()   # simulate on CPU
mybackend = GPU()   # simulate on GPU
solve!(model, nframes=50, backend=mybackend, monitors=mymonitors)


plot_monitors_polarization_xsec("out.hdf", :Ex, :Ey, 1:36)
# inspect("out.hdf", :Ex)
