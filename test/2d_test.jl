grid = Grid2D(
    xmin=-55e-6, xmax=55e-6, Nx=11,
    zmin=-5e-6, zmax=55e-6, Nz=1101,
)

function waveform(x, z, t)
    lam0 = 2e-6   # (m) wavelength
    tau0 = 20e-15   # (s) pulse duration
    dt0 = 3*tau0   # (s) delay time for source injection
    a0 = 10e-6   # (m) beam radius
    w0 = 2*pi * C0 / lam0   # frequency
    return exp(-0.5 * (x / a0)^2) * exp(-(t - dt0)^2 / tau0^2) * cos(w0 * (t - dt0))
end

source = HardSource(
    geometry = (x,z) -> abs(z) < grid.dz/2,
    waveform = waveform,
    component = :Ex,
)

model = Model(grid, source; tmax=150e-15, pml=4e-6)

(; Nx, Nz, x, z) = grid
Eth = zeros(Nx, Nz)
for iz=1:Nz, ix=1:Nx
    Eth[ix,iz] = waveform(x[ix], z[iz], model.t[end] - z[iz]/C0)
end


# CPU:
smodel = solve!(model; fname, backend=CPU())
@test isapprox(smodel.field.Ex, Eth; rtol=5e-2)

if CUDA.functional()
    smodel = solve!(model; fname, backend=GPU())
    @test isapprox(collect(smodel.field.Ex), Eth; rtol=5e-2)
end
