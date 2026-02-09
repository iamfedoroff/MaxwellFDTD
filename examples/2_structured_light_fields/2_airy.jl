# ******************************************************************************************
# Self-accelerating Airy beam
# ******************************************************************************************
using MaxwellFDTD
using MaxwellPlots

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val

import Bessels: airyai


# ******************************************************************************************
# Grid
# ******************************************************************************************
grid = Grid3D(
    xmin=-32e-6, xmax=32e-6, Nx=191,
    ymin=-32e-6, ymax=32e-6, Ny=191,
    zmin=-2e-6, zmax=52e-6, Nz=501,
)


# ******************************************************************************************
# Source
# ******************************************************************************************
lam0 = 1e-6   # (m) central wavelength
tau0 = 30e-15   # (s) FWHM pulse duration

ellipticity = 0   # ellipticity of polarization (ratio of polarization ellipse semi-axes)
theta = deg2rad(0)   # polarization rotation relative to the x-axis

w0 = 2*pi * C0 / lam0   # central frequency

params = (w0, tau0, ellipticity, theta)


function source_geometry(x, y, z)
    return abs(z - 0) <= grid.dz/2
end


function source_waveform_x(x, y, z, t, p)
    w0, tau0, ellipticity, theta = p

    x = x - 5e-6
    wx = wy = 1.5e-6
    ax = ay = 0.1
    xr = x * cos(pi/4) - y * sin(pi/4)
    yr = x * sin(pi/4) + y * cos(pi/4)
    A = airyai(xr / wx) * exp(ax * xr / wx) *
        airyai(yr / wy) * exp(ay * yr / wy)

    # sin^2 temporal amplitude:
    Tx = t > 2*tau0 ? 0.0 : sin(pi * (t - 2*tau0) / (2*tau0))^2 * cos(w0 * (t - 2*tau0))
    Ty = t > 2*tau0 ? 0.0 : sin(pi * (t - 2*tau0) / (2*tau0))^2 * sin(w0 * (t - 2*tau0))

    norm = sqrt(1 + ellipticity^2)
    Ex0 = A * Tx / norm
    Ey0 = ellipticity * A * Ty / norm

    return Ex0 * cos(theta) - Ey0 * sin(theta)
end


function source_waveform_y(x, y, z, t, p)
    w0, tau0, ellipticity, theta = p

    x = x - 5e-6
    wx = wy = 1.5e-6
    ax = ay = 0.1
    xr = x * cos(pi/4) - y * sin(pi/4)
    yr = x * sin(pi/4) + y * cos(pi/4)
    A = airyai(xr / wx) * exp(ax * xr / wx) *
        airyai(yr / wy) * exp(ay * yr / wy)

    # temporal amplitude:
    Tx = t > 2*tau0 ? 0.0 : sin(pi * (t - 2*tau0) / (2*tau0))^2 * cos(w0 * (t - 2*tau0))
    Ty = t > 2*tau0 ? 0.0 : sin(pi * (t - 2*tau0) / (2*tau0))^2 * sin(w0 * (t - 2*tau0))

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
model = Model(grid, source; tmax=230e-15, pml=2e-6)

solve!(model; fname="out.hdf", nframes=50, backend=GPU(), components=(:Ex,))


# ******************************************************************************************
# Visualization
# ******************************************************************************************
# inspect("out.hdf", :Ex; colorrange=(-0.5,0.5))   # field in 3D
# inspect3D_xsec("out.hdf", :Ex)   # field cross-sections
plot3D("out.hdf", :Sa, colorrange=(0,0.1))   # intensity in 3D
# plot3D_xsec("out.hdf", :Sa)   # intensity cross-sections
