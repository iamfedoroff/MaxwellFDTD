# ******************************************************************************************
# Ideal plasmon by a laser pulse propagating along the metal interface.
# The metal is simulated by Drude model with the plasma frequency wp=sqrt(2)*w0, where w0 is
# the central frequency.
# ******************************************************************************************
using MaxwellFDTD
using MaxwellPlots

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val


# ******************************************************************************************
grid = Grid2D(
    xmin=-4e-6, xmax=4e-6, Nx=161,
    zmin=-2e-6, zmax=22e-6, Nz=481,
)


# ******************************************************************************************
lam0 = 2e-6   # (m) central wavelength
tau0 = 20e-15   # (s) pulse duration
dt0 = 5 * tau0   # (s) time shift needed, since the simulation starts at time t=0
w0 = 2*pi * C0 / lam0   # central frequency

params = (w0, tau0, dt0)

function source_waveform(x, z, t, p)
    w0, tau0, dt0 = p
    return exp(-(t - dt0)^2 / tau0^2) * cos(w0 * (t - dt0))
end

function source_geometry(x, z)
    return abs(z - 0) <= grid.dz/2
end

source = SoftSource(
    geometry=source_geometry, waveform=source_waveform, p=params, component=:Ex,
)


# ******************************************************************************************
metal_geometry(x, z) = x >= 0

wp = sqrt(2) * w0
metal_chi = DrudeSusceptibility(wp=wp, gamma=wp/100)
metal = Material(geometry=metal_geometry, chi=metal_chi)

eps0 = 1 + MaxwellFDTD.susceptibility(metal_chi, w0)
n0 = sqrt(eps0)
k0 = n0 * w0 / C0
skin_depth = 1 / (2*imag(k0))
@show n0
@show skin_depth


# ******************************************************************************************
cpml = CPML(thickness=2e-6, kmax=50)

model = Model(grid, source; tmax=800e-15, pml=(2e-6,cpml,2e-6,2e-6), material=metal)

solve!(model, nframes=100, backend=GPU())


# ******************************************************************************************
# plot_geometry(grid.x, grid.z, model.geometry)   # visualize material geometry

inspect("out.hdf", :Ex; colorrange=(-0.5,0.5), aspect=grid.Lx/grid.Lz)
