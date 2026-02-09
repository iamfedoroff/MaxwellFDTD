# ******************************************************************************************
# Interaction of a plane wave with the surface of nickel represented by a Drude-Lorentz
# model.
# For stability reasons, it should be a good enough grid resolution over the skin-depth.
# ******************************************************************************************
using MaxwellFDTD
using MaxwellPlots

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val
const QE = ElementaryCharge.val
const HBAR = ReducedPlanckConstant.val

grid = Grid(
    xmin=-5e-6, xmax=5e-6, Nx=201,
    ymin=-5e-6, ymax=5e-6, Ny=201,
    zmin=-10e-6, zmax=10e-6, Nz=401,
)

function source_geometry(x, y, z)
    return z == -7e-6   # source plane at z=-9um
end

function source_waveform(x, y, z, t)
    lam0 = 2e-6   # (m) central wavelength
    w0 = 2*pi * C0 / lam0   # central frequency
    tau0 = 20e-15   # (s) pulse duration

    A = 1   # plane wave

    # sin2 pulse:
    T = t > 2*tau0 ? 0 : sin(pi * (t - 2*tau0) / (2*tau0))^2 * cos(w0 * (t - 2*tau0))

    return A * T
end

source = SoftSource(geometry=source_geometry, waveform=source_waveform, component=:Ex)

function material_geometry(x, y, z)
    return z >= 0
end


# plot_geometry(grid.x, grid.y, grid.z, material_geometry)   # visualize material geometry


# Nickel from [E. Silaeva et al., Applied Sciences, 11, 21, 9902 (2021)]
# Ni, T=300K (for fitting experimental data):
wp, f0, G0 = 13.28, 0.139, 0.038
fq = [0.107, 0.448, 0.276, 0.673]
Gq = [0.765, 3.007, 2.311, 1.489]
wq = [0.415, 1.439, 4.675, 9.439]

wp = wp / HBAR * QE
G0 = G0 / HBAR * QE
@. Gq = Gq / HBAR * QE
@. wq = wq / HBAR * QE

wpq = sqrt(f0) * wp
gammaq = G0
depsq = @. fq * wp^2 / wq^2
deltaq = @. Gq / 2

Ni_chi = [
    DrudeSusceptibility(wp=wpq, gamma=gammaq),
    LorentzSusceptibility(deps=depsq[1], w0=wq[1], delta=deltaq[1]),
    LorentzSusceptibility(deps=depsq[2], w0=wq[2], delta=deltaq[2]),
    LorentzSusceptibility(deps=depsq[3], w0=wq[3], delta=deltaq[3]),
    LorentzSusceptibility(deps=depsq[4], w0=wq[4], delta=deltaq[4]),
]

Ni = Material(geometry=material_geometry, chi=Ni_chi)

w0 = 2*pi * C0 / 2e-6
n0 = sqrt(1 + MaxwellFDTD.susceptibility(Ni_chi, w0))
ki = imag(n0) * w0 / C0
skin_depth = 1 / (2*ki)
@show skin_depth

model = Model(grid, source; tmax=150e-15, pml=(1e-6,1e-6,1e-6,1e-6,3e-6,3e-6), material=Ni)

# mybackend = CPU()   # simulate on CPU
mybackend = GPU()   # simulate on GPU
solve!(model, nframes=50, backend=mybackend)

inspect("out.hdf", :Ex)
