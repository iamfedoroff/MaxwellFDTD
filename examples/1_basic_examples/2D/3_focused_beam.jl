# ******************************************************************************************
# Example of a focused Gaussian beam.
# ******************************************************************************************
using MaxwellFDTD
using MaxwellPlots

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val

grid = Grid(
    xmin=-50e-6, xmax=50e-6, Nx=501,
    zmin=-50e-6, zmax=50e-6, Nz=501,
)

function source_geometry(x, z)
    return z == -40e-6   # source line at z=-40um
end

function source_waveform(x, z, t)
    lam0 = 2e-6   # (m) central wavelength
    tau0 = 20e-15   # (s) 1/e pulse duration
    dt0 = 5 * tau0   # (s) time shift needed, since the simulation starts at time t=0
    a0 = 10e-6   # (m) beam radius
    zf = 70e-6   # (m) focal distance

    w0 = 2*pi * C0 / lam0   # central frequency
    k0 = w0 / C0   # wave number

    # amplitude:
    # A = 1   # plane wave
    A = exp(-x^2 / a0^2)   # Gaussian beam
    # A = exp(-(x-20e-6)^2 / a0^2) + exp(-(x+20e-6)^2 / a0^2)   # double Gaussian beam

    # phase:
    phi = k0 * x^2 / (2 * zf)
    t = t + phi / w0   # transformation for the amplitude-phase matching

    return A * exp(-(t - dt0)^2 /tau0^2) * cos(w0 * (t - dt0))
end

source = SoftSource(geometry=source_geometry, waveform=source_waveform, component=:Ex)

model = Model(grid, source, tmax=500e-15, pml=5e-6)

# mybackend = CPU()   # simulate on CPU
mybackend = GPU()   # simulate on GPU
solve!(model, nframes=100, backend=mybackend)

inspect("out.hdf", :Ex)
