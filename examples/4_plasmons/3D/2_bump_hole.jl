# ******************************************************************************************
# Excitation of plasmon by a single bump or hole.
# ******************************************************************************************
using MaxwellFDTD
using MaxwellPlots

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val


# ******************************************************************************************
grid = Grid3D(
    xmin=-11e-6, xmax=11e-6, Nx=221,
    ymin=-11e-6, ymax=11e-6, Ny=221,
    zmin=-1e-6, zmax=4e-6, Nz=101,
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
    return abs(z - 3.5e-6) < grid.dz/2
end

source = SoftSource(
    geometry=source_geometry, waveform=source_waveform, p=params, component=:Ex,
)


# ******************************************************************************************
function material_geometry(x, y, z)
    return z <= 0 || sqrt(x^2+y^2+z^2) <= 1e-6   # bump
    # return z <= 0 && ! (sqrt(x^2+y^2+z^2) <= 1e-6)   # hole
end

wp = 2 * w0
metal_chi = DrudeSusceptibility(wp=wp, gamma=wp/100)
metal = Material(geometry=material_geometry, chi=metal_chi)

eps2 = 1 + MaxwellFDTD.susceptibility(metal_chi, w0)
n2 = sqrt(eps2)
k2 = n2 * w0 / C0
skin_depth = 1 / (2*imag(k2))
@show eps2
@show skin_depth


# ******************************************************************************************
cpml = CPML(thickness=1e-6, kmax=20)

model = Model(
    grid, source;
    tmax = 700e-15,
    pml = (cpml, cpml, cpml, cpml, 0.5e-6, 0.5e-6),
    material = metal,
    bc = :dirichlet,
)

solve!(model, nframes=50, backend=GPU())


# ******************************************************************************************
# plot_geometry(grid.x, grid.z, model.geometry; aspect=grid.Lx/grid.Lz)   # visualize material geometry

# inspect("out.hdf", :Ex; colorrange=(-0.01,0.01))
inspect3D_xsec("out.hdf", :Ex; colorrange=(-0.01,0.01), zcut=0)
