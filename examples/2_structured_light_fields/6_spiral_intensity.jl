# ******************************************************************************************
# Spiral intensity distribution using superposition of a focused Gaussian pulse carrying an
# orbital angular momentum and another Gaussian pulse with the flat phase.
# ******************************************************************************************
using MaxwellFDTD
using MaxwellPlots

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val

# adopted from https://discourse.julialang.org/t/laguerre-polynomials-on-gpu/99571/27
function laguerre(x, p::Int, l)
    L0, L1 = 1, 1+l-x
    if p == 0
        return L0
    end
    for k=1:p-1
        L1, L0 = ((2*k+l+1 - x)*L1 - (k+l)*L0)/(k+1), L1
    end
    return L1
end


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
tau0 = 30e-15   # (s) FWHM pulse duration
a0 = 10e-6   # (m) beam radius

ellipticity = 0   # ellipticity of polarization (ratio of polarization ellipse semi-axes)
theta = deg2rad(0)   # polarization rotation relative to the x-axis

l = 1   # topological charge

delta = deg2rad(5)   # focusing angle

w0 = 2*pi * C0 / lam0   # central frequency
k0 = w0 / C0

params = (w0, tau0, a0, ellipticity, theta, l, k0, delta)


function source_geometry(x, y, z)
    return abs(z - 0) <= grid.dz/2
end


function source_waveform_x(x, y, z, t, p)
    w0, tau0, a0, ellipticity, theta, l, k0, delta = p

    r = sqrt(x^2 + y^2)

    # spatial amplitude:
    # A = 1   # plane wave
    A = exp(-0.5 * r^2 / a0^2)   # Gaussian
    # A = (r/a0)^abs(l) * laguerre(r^2/a0^2, 0, abs(l)) * exp(-0.5*r^2/a0^2)   # Laguerre-Gaussian

    # phase:
    phi1 = l * atan(y, x) + k0 * r * sin(delta)
    phi2 = 0
    t1 = t + phi1 / w0   # transformation for the amplitude-phase matching
    t2 = t + phi2 / w0   # transformation for the amplitude-phase matching

    # sin^2 temporal amplitude:
    Tx = t1 > 2*tau0 ? 0.0 : sin(pi * (t1 - 2*tau0) / (2*tau0))^2 * cos(w0 * (t1 - 2*tau0))
    Ty = t1 > 2*tau0 ? 0.0 : sin(pi * (t1 - 2*tau0) / (2*tau0))^2 * sin(w0 * (t1 - 2*tau0))

    norm = sqrt(1 + ellipticity^2)
    Ex0 = A * Tx / norm
    Ey0 = ellipticity * A * Ty / norm

    # rotate:
    Ex = Ex0 * cos(theta) - Ey0 * sin(theta)

    # second pulse:
    Tx = t2 > 2*tau0 ? 0.0 : sin(pi * (t2 - 2*tau0) / (2*tau0))^2 * cos(w0 * (t2 - 2*tau0))
    Ty = t2 > 2*tau0 ? 0.0 : sin(pi * (t2 - 2*tau0) / (2*tau0))^2 * sin(w0 * (t2 - 2*tau0))
    Ex0 = A * Tx / norm
    Ey0 = ellipticity * A * Ty / norm
    Ex2 = Ex0 * cos(theta) - Ey0 * sin(theta)

    return (Ex + Ex2) / sqrt(2)
end


function source_waveform_y(x, y, z, t, p)
    w0, tau0, a0, ellipticity, theta, l, k0, delta = p

    r = sqrt(x^2 + y^2)

    # spatial amplitude:
    # A = 1   # plane wave
    A = exp(-0.5 * r^2 / a0^2)   # Gaussian
    # A = (r/a0)^abs(l) * laguerre(r^2/a0^2, 0, abs(l)) * exp(-0.5*r^2/a0^2)   # Laguerre-Gaussian

    # phase:
    phi1 = l * atan(y, x) + k0 * r * sin(delta)
    phi2 = 0
    t1 = t + phi1 / w0   # transformation for the amplitude-phase matching
    t2 = t + phi2 / w0   # transformation for the amplitude-phase matching

    # sin^2 temporal amplitude:
    Tx = t1 > 2*tau0 ? 0.0 : sin(pi * (t1 - 2*tau0) / (2*tau0))^2 * cos(w0 * (t1 - 2*tau0))
    Ty = t1 > 2*tau0 ? 0.0 : sin(pi * (t1 - 2*tau0) / (2*tau0))^2 * sin(w0 * (t1 - 2*tau0))

    norm = sqrt(1 + ellipticity^2)
    Ex0 = A * Tx / norm
    Ey0 = ellipticity * A * Ty / norm

    # rotate:
    Ey = Ex0 * sin(theta) + Ey0 * cos(theta)

    # second pulse:
    Tx = t2 > 2*tau0 ? 0.0 : sin(pi * (t2 - 2*tau0) / (2*tau0))^2 * cos(w0 * (t2 - 2*tau0))
    Ty = t2 > 2*tau0 ? 0.0 : sin(pi * (t2 - 2*tau0) / (2*tau0))^2 * sin(w0 * (t2 - 2*tau0))
    Ex0 = A * Tx / norm
    Ey0 = ellipticity * A * Ty / norm
    Ey2 = Ex0 * sin(theta) + Ey0 * cos(theta)

    return (Ey + Ey2) / sqrt(2)
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

solve!(model; fname="out.hdf", nframes=50, backend=GPU(), components=(:Ex,))


# ******************************************************************************************
# Visualization
# ******************************************************************************************
# inspect("out.hdf", :Ex; colorrange=(-0.5,0.5))   # field in 3D
# inspect3D_xsec("out.hdf", :Ex)   # field cross-sections
plot3D("out.hdf", :Sa)   # intensity in 3D
# plot3D_xsec("out.hdf", :Sa)   # intensity cross-sections
