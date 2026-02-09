# ******************************************************************************************
# Excitation of plasmon by a metallic grating.
# ******************************************************************************************
using MaxwellFDTD
using MaxwellPlots

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val


# ******************************************************************************************
grid = Grid2D(
    xmin=-30e-6, xmax=60e-6, Nx=1801,
    zmin=-7e-6, zmax=17e-6, Nz=961,
)


# ******************************************************************************************
lam0 = 2e-6   # (m) central wavelength
tau0 = 20e-15   # (s) pulse duration
dt0 = 5 * tau0   # (s) time shift needed, since the simulation starts at time t=0
a0 = 10e-6   # (m) beam radius

n0 = 1
theta = deg2rad(13)   # lamg = 2e-6
# theta = deg2rad(33.9)  # lamg = 3e-6

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
    return abs(z - 15e-6) < grid.dz/2
end

source_x = SoftSource(
    geometry=source_geometry, waveform=source_waveform_x, p=params, component=:Ex,
)
source_z = SoftSource(
    geometry=source_geometry, waveform=source_waveform_z, p=params, component=:Ez,
)
source = (source_x, source_z)


# ******************************************************************************************
function material_geometry(x, z)
    thickness = 500e-9

    A = 500e-9
    lamg = 2e-6

    # sinusoidal grating:
    kg = 2*pi / lamg
    G = z <= A * sin(kg * x)

    # square grating:
    # G = z <= -A
    # for i=-30:60
    #     G = G || (z <= A && abs(x-i*lamg) <= lamg/4)
    # end

    return G && z >= -(A + thickness)
end


wp = 2 * w0
metal_chi = DrudeSusceptibility(wp=wp, gamma=wp/100)
metal = Material(geometry=material_geometry, chi=metal_chi)


eps1 = n0^2
eps2 = 1 + MaxwellFDTD.susceptibility(metal_chi, w0)
n2 = sqrt(eps2)
k2 = n2 * w0 / C0
skin_depth = 1 / (2*imag(k2))

kx = k0 * sin(theta)

lamg = 2e-6
kg = n0 * 2*pi / lamg
ksp = sqrt(eps1 * eps2 / (eps1 + eps2) + 0im) * w0 / C0

th = asin((real(ksp) - kg) / k0)
# ksp_p1 = k0 * sin(theta) + kg
# ksp_m1 = k0 * sin(theta) - kg

@show eps2
@show skin_depth
# @show ksp_p1 / k0
# @show ksp_m1 / k0
# @show (real(ksp) - kg) / k0
# @show rad2deg(th)
@show kx / (w0/C0)
@show kg / (w0/C0)
@show (kx + kg) / (w0/C0)
@show real(ksp) / (w0/C0)
@show rad2deg(th)


# ******************************************************************************************
cpml = CPML(thickness=4e-6, kmax=12)

model = Model(
    grid, source;
    tmax = 600e-15,
    pml = (cpml, cpml, 2e-6, 2e-6),
    material = metal,
    bc = :dirichlet,
)

solve!(model, nframes=100, backend=GPU())


# ******************************************************************************************
# plot_geometry(grid.x, grid.z, model.geometry)   # visualize material geometry

inspect("out.hdf", :Ex; colorrange=(-0.01,0.01), aspect=grid.Lx/grid.Lz)
