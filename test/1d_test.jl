grid = Grid1D(zmin=-5e-6, zmax=55e-6, Nz=2001)

lam0 = 2e-6   # (m) wavelength
tau0 = 20e-15   # (s) pulse duration
dt0 = 3*tau0   # (s) delay time for source injection
w0 = 2*pi * C0 / lam0   # frequency

function waveform(t, p)
    w0, tau0, dt0 = p
    return exp(-(t - dt0)^2 / tau0^2) * cos(w0 * (t - dt0))
end

amplitude(z) = 1

source = HardSource(
    geometry = z -> abs(z) < grid.dz/2,
    amplitude = amplitude,
    waveform = waveform,
    p = (w0,tau0,dt0),
    component = :Ex,
)

model = Model(grid, source; tmax=150e-15, pml_box=(4e-6,4e-6))

Eth = @. waveform(model.t[end] - grid.z/C0, ((w0,tau0,dt0),))


# CPU:
smodel = solve!(model; fname, arch=CPU())
@test isapprox(smodel.field.Ex, Eth; rtol=1e-2)

# GPU:
if CUDA.functional()
    smodel = solve!(model; fname, arch=GPU())
    @test isapprox(collect(smodel.field.Ex), Eth; rtol=1e-2)
end
