# ******************************************************************************************
# Plasmon excitation using the Otto configuration.
# Oblique incident pulse, normal interfaces.
# ******************************************************************************************
using MaxwellFDTD
using MaxwellPlots

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val


# ******************************************************************************************
grid = Grid2D(
    xmin=-30e-6, xmax=70e-6, Nx=1001,
    zmin=-4e-6, zmax=22e-6, Nz=261,
)


# ******************************************************************************************
lam0 = 2e-6   # (m) central wavelength
tau0 = 20e-15   # (s) pulse duration
dt0 = 5 * tau0   # (s) time shift needed, since the simulation starts at time t=0
a0 = 10e-6   # (m) beam radius

n0 = 2   # prism refractive index
theta = asin(1/n0) + deg2rad(5)   # incidence angle

w0 = 2*pi * C0 / lam0   # central frequency
k0 = n0 * w0 / C0   # wave vector

params = (w0, tau0, dt0, a0, theta, k0)

function source_waveform_x(x, z, t, p)
    w0, tau0, dt0, a0, theta, k0 = p
    A = cos(theta) * exp(-x^2 / a0^2)
    phase = k0 * tan(theta) * x
    t = t - phase / w0   # apply phase through the temporal delay
    return A * exp(-(t - dt0)^2 /tau0^2) * cos(w0 * (t - dt0))
end

function source_waveform_z(x, z, t, p)
    w0, tau0, dt0, a0, theta, k0 = p
    A = sin(theta) * exp(-x^2 / a0^2)
    phase = k0 * tan(theta) * x
    t = t - phase / w0   # apply phase through the temporal delay
    return A * exp(-(t - dt0)^2 /tau0^2) * cos(w0 * (t - dt0))
end

function source_geometry(x, z)
    return abs(z - 20e-6) <= grid.dz/2
end

source_x = SoftSource(
    geometry=source_geometry, waveform=source_waveform_x, p=params, component=:Ex,
)
source_z = SoftSource(
    geometry=source_geometry, waveform=source_waveform_z, p=params, component=:Ez,
)
source = (source_x, source_z)


# ******************************************************************************************
glass_geometry(x, z) = z >= 0
glass = Material(geometry=glass_geometry, eps=n0^2)

wp = 2 * w0
metal_chi = DrudeSusceptibility(wp=wp, gamma=wp/100)
metal_geometry(x, z) = z <= -1e-6
metal = Material(geometry=metal_geometry, chi=metal_chi)

eps2 = 1 + MaxwellFDTD.susceptibility(metal_chi, w0)
n2 = sqrt(eps2)
k2 = n2 * w0 / C0
skin_depth = 1 / (2*imag(k2))
@show eps2
@show skin_depth


# ******************************************************************************************
cpml = CPML(thickness=2e-6, kmax=10)

model = Model(
    grid, source;
    tmax = 700e-15,
    pml = (cpml, cpml, 2e-6, 2e-6),
    material = (glass, metal),
    bc = :dirichlet,
)

solve!(model, nframes=100, backend=GPU())


# ******************************************************************************************
# plot_geometry(grid.x, grid.z, model.geometry)   # visualize material geometry

inspect("out.hdf", :Ex; colorrange=(-0.1,0.1), aspect=grid.Lx/grid.Lz)
