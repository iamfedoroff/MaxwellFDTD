fname = joinpath(@__DIR__, "out.hdf")

grid = Grid1D(zmin=-5e-6, zmax=35e-6, Nz=2001)

lam0 = 2e-6   # (m) wavelength
tau0 = 20e-15   # (s) pulse duration
dt0 = 3*tau0   # (s) delay time for source injection
w0 = 2*pi * C0 / lam0   # frequency

function waveform(t, p)
    w0, tau0, dt0 = p
    return exp(-(t - dt0)^2 / tau0^2) * cos(w0 * (t - dt0))
end

source = HardSource(
    geometry = z -> abs(z) < grid.dz/2,
    amplitude = z -> 1,
    waveform = waveform,
    p = (w0,tau0,dt0),
    component=:Ex,
)

model = Model(grid, source; tmax=230e-15, pml_box=(4e-6,4e-6))

solve!(model; fname, viewpoints=(30e-6,))


fp = HDF5.h5open(fname, "r")
t = HDF5.read(fp, "viewpoints/t")
z = HDF5.read(fp, "viewpoints/1/point")[1]
Ex = HDF5.read(fp, "viewpoints/1/Ex")
HDF5.close(fp)
rm(fname)

Eth = @. waveform(t - z/C0, ((w0,tau0,dt0),))

@test isapprox(Ex, Eth; rtol=2e-2)
