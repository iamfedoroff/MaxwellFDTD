grid = Grid3D(
    xmin=-55e-6, xmax=55e-6, Nx=11,
    ymin=-55e-6, ymax=55e-6, Ny=11,
    zmin=-5e-6, zmax=55e-6, Nz=801,
)

function waveform(x, y, z, t)
    lam0 = 2e-6   # (m) wavelength
    tau0 = 20e-15   # (s) pulse duration
    dt0 = 3*tau0   # (s) delay time for source injection
    a0 = 10e-6   # (m) beam radius
    w0 = 2*pi * C0 / lam0   # frequency
    return exp(-0.5 * (sqrt(x^2 + y^2) / a0)^2) *
           exp(-(t - dt0)^2 / tau0^2) * cos(w0 * (t - dt0))
end

source = HardSource(
    geometry = (x,y,z) -> abs(z) < grid.dz/2,
    waveform = waveform,
    component = :Ex,
)

model = Model(grid, source; tmax=150e-15, pml_box=(4e-6,4e-6,4e-6,4e-6,4e-6,4e-6))

(; Nx, Ny, Nz, x, y, z) = grid
Eth = zeros(Nx, Ny, Nz)
for iz=1:Nz, iy=1:Ny, ix=1:Nx
    Eth[ix,iy,iz] = waveform(x[ix], y[iy], z[iz], model.t[end] - z[iz]/C0)
end


# CPU:
smodel = solve!(model; fname, arch=CPU())
@test isapprox(smodel.field.Ex, Eth; rtol=1e-1)

if CUDA.functional()
    smodel = solve!(model; fname, arch=GPU())
    @test isapprox(collect(smodel.field.Ex), Eth; rtol=1e-1)
end
