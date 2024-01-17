function test_intensity_losses_1D(fname, ki)
    fp = HDF5.h5open(fname, "r")
    z = HDF5.read(fp, "z")
    Sa = HDF5.read(fp, "Sa")
    HDF5.close(fp)

    Nz = length(Sa)

    mask = @. z < 0 || z > 20e-6

    Sa = [mask[iz] ? 0.0 : Sa[iz] for iz=1:Nz]
    Sa .= Sa ./ maximum(Sa)

    I = [mask[iz] ? 0.0 : exp(-2 * ki * z[iz]) for iz=1:Nz]

    return isapprox(Sa, I; rtol=1e-2)
end


# ******************************************************************************************
grid = Grid1D(zmin=-15e-6, zmax=25e-6, Nz=2001)

function source_geometry(z)
    return abs(z - -10e-6) <= grid.dz/2
end

function source_waveform(z, t)
    lam0 = 2e-6   # (m) wavelength
    tau0 = 20e-15   # (s) pulse duration
    dt0 = 3*tau0   # (s) delay time for source injection
    w0 = 2*pi * C0 / lam0   # frequency
    return exp(-(t - dt0)^2 / tau0^2) * cos(w0 * (t - dt0))
end

source = SoftSource(geometry=source_geometry, waveform=source_waveform, component=:Ex)

function material_geometry(z)
    return z >= 0   # material occupies half-space z>0
end

lam0 = 2e-6   # (m) central wavelength
w0 = 2*pi * C0 / lam0   # central frequency


# ******************************************************************************************
# Conductivity losses
# ******************************************************************************************
sigma = 300   # conductivity
er = 1.5^2   # real part of permittivity
ei = sigma / (EPS0*w0)   # imaginary part of permittivity
ni = sqrt((sqrt(er^2 + ei^2) - er) / 2)   # imaginary part of refractive index
ki = ni * w0 / C0   # imaginary part of wavevector

material = Material(geometry=material_geometry, eps=er, sigma=sigma)

model = Model(grid, source; tmax=300e-15, pml=5e-6, material)

solve!(model; fname)
@test test_intensity_losses_1D(fname, ki)

if CUDA.functional()
    solve!(model; fname, backend=GPU())
    @test test_intensity_losses_1D(fname, ki)
end


# ******************************************************************************************
# Dispersion losses
# ******************************************************************************************
depsq = 2
wq = 2*pi * C0 / 1e-6
deltaq = 0.01 * wq
chi = LorentzSusceptibility(deps=depsq, w0=wq, delta=deltaq)

n0 = sqrt(1 + MaxwellFDTD.susceptibility(chi, w0))   # refractive index
ki = imag(n0) * w0 / C0   # imaginary part of wavevector

material = Material(geometry=material_geometry, chi=chi)

model = Model(grid, source; tmax=600e-15, pml=5e-6, material)

solve!(model; fname)
@test test_intensity_losses_1D(fname, ki)

if CUDA.functional()
    solve!(model; fname, backend=GPU())
    @test test_intensity_losses_1D(fname, ki)
end
