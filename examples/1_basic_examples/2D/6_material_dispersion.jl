# ******************************************************************************************
# Interaction of a plane wave with the surface of gold represented by a Drude-Lorentz model.
# For stability reasons, it should be a good enough grid resolution over the skin-depth.
# ******************************************************************************************
using MaxwellFDTD
using MaxwellPlots

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val
const QE = ElementaryCharge.val
const HBAR = ReducedPlanckConstant.val

grid = Grid(
    xmin=-2e-6, xmax=2e-6, Nx=801,
    zmin=-0.2e-6, zmax=0.5e-6, Nz=141,
)

function source_geometry(x, z)
    return abs(z - 0.45e-6) <= grid.dz/2
end

function source_waveform(x, z, t)
    lam0 = 2e-6   # (m) central wavelength
    w0 = 2*pi * C0 / lam0   # central frequency
    tau0 = 30e-15   # (s) FWHM pulse duration

    A = 1   # plane wave

    # sin2 pulse:
    T = t > 2*tau0 ? 0 : sin(pi * (t - 2*tau0) / (2*tau0))^2 * cos(w0 * (t - 2*tau0))

    return A * T
end

source = SoftSource(geometry=source_geometry, waveform=source_waveform, component=:Ex)

function material_geometry(x, z)
    return z <= 0
end


# plot_geometry(grid.x, grid.z, material_geometry)   # visualize material geometry


# Gold from [E. Silaeva et al., Applied Sciences, 11, 21, 9902 (2021)]
# Au, T=300K (for fitting experimental data):
wp, f0, G0 = 13.98, 0.144, 0.001
fq = [0.192, 0.026, 0.168, 0.744]
Gq = [0.125, 0.779, 2.136, 4.685]
wq = [0.010, 3.003, 4.134, 7.586]

wp = wp / HBAR * QE
G0 = G0 / HBAR * QE
@. Gq = Gq / HBAR * QE
@. wq = wq / HBAR * QE

wpq = sqrt(f0) * wp
gammaq = G0
depsq = @. fq * wp^2 / wq^2
deltaq = @. Gq / 2

Au_chi = [
    DrudeSusceptibility(wp=wpq, gamma=gammaq),
    LorentzSusceptibility(deps=depsq[1], w0=wq[1], delta=deltaq[1]),
    LorentzSusceptibility(deps=depsq[2], w0=wq[2], delta=deltaq[2]),
    LorentzSusceptibility(deps=depsq[3], w0=wq[3], delta=deltaq[3]),
    LorentzSusceptibility(deps=depsq[4], w0=wq[4], delta=deltaq[4]),
]

Au = Material(geometry=material_geometry, chi=Au_chi)

w0 = 2*pi * C0 / 2e-6
n0 = sqrt(1 + MaxwellFDTD.susceptibility(Au_chi, w0))
ki = imag(n0) * w0 / C0
skin_depth = 1 / (2*ki)
@show skin_depth


model = Model(grid, source; tmax=250e-15, pml=0.05e-6, material=Au)

# mybackend = CPU()   # simulate on CPU
mybackend = GPU()   # simulate on GPU
solve!(model, nframes=100, backend=mybackend)

inspect("out.hdf", :Ex)
