grid = Grid2D(
    xmin=-55e-6, xmax=55e-6, Nx=11,
    zmin=-5e-6, zmax=55e-6, Nz=1101,
)

lam0 = 2e-6   # (m) wavelength
tau0 = 20e-15   # (s) pulse duration
dt0 = 3*tau0   # (s) delay time for source injection
a0 = 10e-6   # (m) beam radius
w0 = 2*pi * C0 / lam0   # frequency

function waveform(t, p)
    w0, tau0, dt0 = p
    return exp(-(t - dt0)^2 / tau0^2) * cos(w0 * (t - dt0))
end

amplitude(x,z) = exp(-0.5 * (x / a0)^2)

source = HardSource(
    geometry = (x,z) -> abs(z) < grid.dz/2,
    amplitude = amplitude,
    waveform = waveform,
    p = (w0,tau0,dt0),
    component = :Ex,
)

model = Model(grid, source; tmax=150e-15, pml_box=(4e-6,4e-6,4e-6,4e-6))

Eth = zeros(grid.Nx, grid.Nz)
for iz=1:grid.Nz, ix=1:grid.Nx
    A = amplitude(grid.x[ix], grid.z[iz])
    T = waveform(model.t[end] - grid.z[iz]/C0, (w0,tau0,dt0))
    Eth[ix,iz] = A * T
end


# CPU:
smodel = solve!(model; fname, arch=CPU())
@test isapprox(smodel.field.Ex, Eth; rtol=5e-2)

if CUDA.functional()
    smodel = solve!(model; fname, arch=GPU())
    @test isapprox(collect(smodel.field.Ex), Eth; rtol=5e-2)
end
