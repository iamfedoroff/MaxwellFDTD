# ******************************************************************************************
# Plasmon excitation using the Otto configuration.
# Normal incident pulse, oblique interfaces.
# ******************************************************************************************
using MaxwellFDTD
using MaxwellPlots

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val


# ******************************************************************************************
grid = Grid2D(
    xmin=-30e-6, xmax=35e-6, Nx=651,
    zmin=-35e-6, zmax=12e-6, Nz=471,
)


# ******************************************************************************************
lam0 = 2e-6   # (m) central wavelength
tau0 = 20e-15   # (s) pulse duration
dt0 = 5 * tau0   # (s) time shift needed, since the simulation starts at time t=0
a0 = 10e-6   # (m) beam radius

n0 = 2   # prism refractive index

w0 = 2*pi * C0 / lam0   # central frequency

params = (w0, tau0, dt0, a0)

function source_waveform(x, z, t, p)
    w0, tau0, dt0, a0 = p
    A = exp(-x^2 / a0^2)   # Gaussian beam
    return A * exp(-(t - dt0)^2 /tau0^2) * cos(w0 * (t - dt0))
end

function source_geometry(x, z)
    return abs(z - 10e-6) < grid.dz/2
end

source = SoftSource(
    geometry=source_geometry, waveform=source_waveform, p=params, component=:Ex,
)


# ******************************************************************************************
function glass_geometry(x, z)
    n0 = 2   # prism refractive index
    theta = asin(1/n0) + deg2rad(5)   # incidence angle
    return z > -tan(theta) * x - 10e-6
end
glass = Material(geometry=glass_geometry, eps=n0^2)


wp = 2 * w0
metal_chi = DrudeSusceptibility(wp=wp, gamma=wp/100)
function metal_geometry(x, z)
    n0 = 2   # prism refractive index
    theta = asin(1/n0) + deg2rad(5)   # incidence angle
    return z <= -tan(theta) * x - 10e-6 - 1e-6
end
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
