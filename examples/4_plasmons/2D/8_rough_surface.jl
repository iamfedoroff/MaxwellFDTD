# ******************************************************************************************
# Excitation of plasmon at a rough surface.
# ******************************************************************************************
using MaxwellFDTD
using MaxwellPlots
using RoughEdges

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val


# ******************************************************************************************
grid = Grid2D(
    xmin=-50e-6, xmax=50e-6, Nx=2001,
    zmin=-3e-6, zmax=7e-6, Nz=201,
)


# ******************************************************************************************
lam0 = 2e-6   # (m) central wavelength
tau0 = 20e-15   # (s) pulse duration
dt0 = 5 * tau0   # (s) time shift needed, since the simulation starts at time t=0

w0 = 2*pi * C0 / lam0   # central frequency

params = (w0, tau0, dt0)

function source_geometry(x, z)
    return abs(z - 5e-6) < grid.dz/2
end

function source_waveform(x, z, t, p)
    w0, tau0, dt0 = p
    return exp(-(t - dt0)^2 /tau0^2) * cos(w0 * (t - dt0))
end

source = SoftSource(
    geometry=source_geometry, waveform=source_waveform, p=params, component=:Ex,
)


# ******************************************************************************************
sigma = 500e-9
xi = 1e-6

R = rough(grid.x; sigma, xi, seed=1)
gmask = geometry_mask(grid.x, grid.z, R)

# rough_plot(grid.x, R)
# plot_geometry(grid.x, grid.z, gmask; aspect=grid.Lx/grid.Lz)

wp = 2 * w0
metal_chi = DrudeSusceptibility(wp=wp, gamma=wp/100)
metal = Material(geometry=gmask, chi=metal_chi)

eps2 = 1 + MaxwellFDTD.susceptibility(metal_chi, w0)
n2 = sqrt(eps2)
k2 = n2 * w0 / C0
skin_depth = 1 / (2*imag(k2))
@show eps2
@show skin_depth


# ******************************************************************************************
cpml = CPML(thickness=5e-6, kmax=20)

model = Model(
    grid, source;
    tmax = 1000e-15,
    pml=(cpml, cpml, 1e-6, 2e-6),
    material = metal,
    bc = :dirichlet,
)

solve!(model, nframes=100, backend=GPU())


# ******************************************************************************************
# plot_geometry(grid.x, grid.z, model.geometry; aspect=grid.Lx/grid.Lz)

inspect("out.hdf", :Ex; colorrange=(-0.1,0.1), aspect=grid.Lx/grid.Lz)
