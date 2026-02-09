# ******************************************************************************************
# Space-time optical vortex
#
# A.P. Sukhorukov and V.V. Yangirova "Spatio-temporal vortices: properties, generation
# and recording", Proc. SPIE 5949, Nonlinear Optics Applications, 594906 (2005)
# https://doi.org/10.1117/12.623906
# ******************************************************************************************
using MaxwellFDTD
using MaxwellPlots

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val


# ******************************************************************************************
# Grid
# ******************************************************************************************
grid = Grid3D(
    xmin=-42e-6, xmax=42e-6, Nx=211,
    ymin=-42e-6, ymax=42e-6, Ny=211,
    zmin=-2e-6, zmax=22e-6, Nz=241,
)


# ******************************************************************************************
# Source
# ******************************************************************************************
lam0 = 1e-6   # (m) central wavelength
tau0 = 20e-15   # (s) pulse duration
a0 = 10e-6   # (m) beam radius

ellipticity = 0   # ellipticity of polarization (ratio of polarization ellipse semi-axes)
theta = deg2rad(0)   # polarization rotation relative to the x-axis

l = 1   # topological charge

w0 = 2*pi * C0 / lam0   # central frequency

params = (w0, tau0, a0, ellipticity, theta, l)


function source_geometry(x, y, z)
    return abs(z - 0) <= grid.dz/2
end


function source_waveform_x(x, y, z, t, p)
    w0, tau0, a0, ellipticity, theta, l = p

    r = sqrt(x^2 + y^2)

    # spatial amplitude:
    A = exp(-0.5 * r^2 / a0^2)

    # stov phase:
    phi = l * atan(t/tau0, x/a0)
    # phi = l * atan(t/tau0, r/a0)
    t = t + phi / w0   # transformation for the amplitude-phase matching

    # sin^2 temporal amplitude:
    Tx = t > 2*tau0 ? 0.0 : sin(pi * (t - 2*tau0) / (2*tau0))^2 * cos(w0 * (t - 2*tau0))
    Ty = t > 2*tau0 ? 0.0 : sin(pi * (t - 2*tau0) / (2*tau0))^2 * sin(w0 * (t - 2*tau0))

    # hole in x-t coordinates:
    A = A * sqrt(((t - 2*tau0/2)/tau0)^2 + (x/a0)^2)^abs(l)
    # A = A * sqrt(((t - 2*tau0/2)/tau0)^2 + (r/a0)^2)^abs(l)

    norm = sqrt(1 + ellipticity^2)
    Ex0 = A * Tx / norm
    Ey0 = ellipticity * A * Ty / norm

    return Ex0 * cos(theta) - Ey0 * sin(theta)
end


function source_waveform_y(x, y, z, t, p)
    w0, tau0, a0, ellipticity, theta, l = p

    r = sqrt(x^2 + y^2)

    # spatial amplitude:
    A = exp(-0.5 * r^2 / a0^2)

    # stov phase:
    # phi = l * atan(t/tau0, x/a0)
    phi = l * atan(t/tau0, r/a0)
    t = t + phi / w0   # transformation for the amplitude-phase matching

    # sin^2 temporal amplitude:
    Tx = t > 2*tau0 ? 0.0 : sin(pi * (t - 2*tau0) / (2*tau0))^2 * cos(w0 * (t - 2*tau0))
    Ty = t > 2*tau0 ? 0.0 : sin(pi * (t - 2*tau0) / (2*tau0))^2 * sin(w0 * (t - 2*tau0))

    # hole in x-t coordinates:
    A = A * sqrt(((t - 2*tau0/2)/tau0)^2 + (x/a0)^2)^abs(l)
    # A = A * sqrt(((t - 2*tau0/2)/tau0)^2 + (r/a0)^2)^abs(l)

    norm = sqrt(1 + ellipticity^2)
    Ex0 = A * Tx / norm
    Ey0 = ellipticity * A * Ty / norm

    return Ex0 * sin(theta) + Ey0 * cos(theta)
end


source_x = SoftSource(
    geometry=source_geometry, waveform=source_waveform_x, p=params, component=:Ex,
)

source_y = SoftSource(
    geometry=source_geometry, waveform=source_waveform_y, p=params, component=:Ey,
)

source = (source_x, source_y)


# ******************************************************************************************
# Model & Solve
# ******************************************************************************************
model = Model(grid, source; tmax=180e-15, pml=2e-6)

solve!(model; fname="out.hdf", nframes=50, backend=GPU())


# ******************************************************************************************
# Visualization
# ******************************************************************************************
# inspect("out.hdf", :Ex; colorrange=(-0.5,0.5))   # field in 3D
inspect3D_xsec("out.hdf", :Ex)   # field cross-sections
# plot3D("out.hdf", :Sa)   # intensity in 3D
# plot3D_xsec("out.hdf", :Sa)   # intensity cross-sections
