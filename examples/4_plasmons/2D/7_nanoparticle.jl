# ******************************************************************************************
# Excitation of plasmon at a single nanoparticle.
# ******************************************************************************************
using MaxwellFDTD
using MaxwellPlots

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val


# ******************************************************************************************
grid = Grid2D(
    xmin=-5e-6, xmax=5e-6, Nx=1001,
    zmin=-5e-6, zmax=5e-6, Nz=1001,
)


# ******************************************************************************************
lam0 = 2e-6   # (m) central wavelength
tau0 = 20e-15   # (s) pulse duration
dt0 = 5 * tau0   # (s) time shift needed, since the simulation starts at time t=0
a0 = 10e-6   # (m) beam radius

w0 = 2*pi * C0 / lam0   # central frequency

params = (w0, tau0, dt0, a0)


function source_geometry(x, z)
    return abs(z - 4e-6) < grid.dz/2
end

function source_waveform(x, z, t, p)
    w0, tau0, dt0, a0 = p
    A = 1
    # A = cos(theta) * exp(-x^2 / a0^2)   # Gaussian beam
    return A * exp(-(t - dt0)^2 /tau0^2) * cos(w0 * (t - dt0))
end

source = SoftSource(
    geometry=source_geometry, waveform=source_waveform, p=params, component=:Ex,
)


# ******************************************************************************************
function material_geometry(x, z)
    return sqrt(x^2 + z^2) <= 1e-6
end

glass = Material(geometry=material_geometry, eps=2^2)

wp = 1.7323 * w0
metal_chi = DrudeSusceptibility(wp=wp, gamma=wp/100)
metal = Material(geometry=material_geometry, chi=metal_chi)

eps2 = 1 + MaxwellFDTD.susceptibility(metal_chi, w0)
n2 = sqrt(eps2)
k2 = n2 * w0 / C0
skin_depth = 1 / (2*imag(k2))
@show eps2
@show skin_depth


# ******************************************************************************************
model = Model(grid, source; tmax=500e-15, pml=1e-6, material = metal)

solve!(model, nframes=100, backend=GPU())


# ******************************************************************************************
# plot_geometry(grid.x, grid.z, model.geometry; aspect=grid.Lx/grid.Lz)   # visualize material geometry

inspect("out.hdf", :Ex; colorrange=(-0.1,0.1), aspect=grid.Lx/grid.Lz)
