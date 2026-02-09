# ******************************************************************************************
# Excitation of plasmon by a dipole located near to a metallic surface.
# ******************************************************************************************
using MaxwellFDTD
using MaxwellPlots

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val


# ******************************************************************************************
grid = Grid3D(
    xmin=-12e-6, xmax=12e-6, Nx=241,
    ymin=-12e-6, ymax=12e-6, Ny=241,
    zmin=-4e-6, zmax=6e-6, Nz=101,
)


# ******************************************************************************************
lam0 = 2e-6   # (m) central wavelength
tau0 = 20e-15   # (s) pulse duration
dt0 = 5 * tau0   # (s) time shift needed, since the simulation starts at time t=0

w0 = 2*pi * C0 / lam0   # central frequency

params = (w0, tau0, dt0)

function source_waveform(x, y, z, t, p)
    w0, tau0, dt0 = p
    return exp(-(t - dt0)^2 /tau0^2) * cos(w0 * (t - dt0))
end

function source_geometry(x, y, z)
    return abs(x - 0) <= grid.dx/2 &&
           abs(y - 0) <= grid.dy/2 &&
           abs(z - lam0/2) <= grid.dz/2
end

source = SoftSource(
    geometry=source_geometry, waveform=source_waveform, p=params, component=:Ex,
)


# ******************************************************************************************
wp = 2 * w0
metal_chi = DrudeSusceptibility(wp=wp, gamma=wp/100)
metal_geometry(x, y, z) = z <= 0
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
    tmax = 500e-15,
    pml = (cpml, cpml, cpml, cpml, 2e-6, 2e-6),
    material = metal,
    bc = :dirichlet,
)

solve!(model, nframes=50, backend=GPU())


# ******************************************************************************************
# plot_geometry(grid.x, grid.z, model.geometry)   # visualize material geometry

inspect("out.hdf", :Ex; colorrange=(-1e-3,1e-3))
# inspect3D_xsec("out.hdf", :Ex; colorrange=(-1e-3,1e-3), zcut=0)
