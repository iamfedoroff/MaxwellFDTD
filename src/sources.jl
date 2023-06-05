abstract type Source end


# ******************************************************************************************
# Soft
# ******************************************************************************************
struct DataSoftSource{FG, FA, FP, FW, P, T, S}
    geometry :: FG
    amplitude :: FA
    phase :: FP
    waveform :: FW
    p :: P
    frequency :: T
    component :: S
end


struct SoftSource{C, A, F, P} <: Source
    isrc :: C
    Amp :: A
    Phi :: A
    waveform :: F
    p :: P
    icomp :: Int
end

@adapt_structure SoftSource


function SoftSource(; geometry, amplitude, phase, waveform, p, frequency, component)
    return DataSoftSource(geometry, amplitude, phase, waveform, p, frequency, component)
end


function source_init(data::DataSoftSource, field::Field1D, t)
    (; geometry, amplitude, phase, waveform, p, frequency, component) = data
    (; grid) = field
    (; z) = grid

    geom = @. geometry(z)
    isrc = findall(geom)

    Amp, Phi = zeros(size(isrc)), zeros(size(isrc))
    for (i, iz) in enumerate(isrc)
        Amp[i] = amplitude(z[iz])
        Phi[i] = phase(z[iz]) / frequency
    end

    icomp = findfirst(isequal(component), fieldnames(typeof(field)))   # Symbol -> Int

    return SoftSource(isrc, Amp, Phi, waveform, p, icomp)
end


function source_init(data::DataSoftSource, field::Field2D, t)
    (; geometry, amplitude, phase, waveform, p, frequency, component) = data
    (; grid) = field
    (; Nx, Nz, x, z) = grid

    geom = [geometry(x[ix], z[iz]) for ix=1:Nx, iz=1:Nz]
    isrc = findall(geom)

    Amp, Phi = zeros(size(isrc)), zeros(size(isrc))
    for (i, ici) in enumerate(isrc)
        ix, iz = ici[1], ici[2]
        Amp[i] = amplitude(x[ix], z[iz])
        Phi[i] = phase(x[ix], z[iz]) / frequency
    end

    icomp = findfirst(isequal(component), fieldnames(typeof(field)))   # Symbol -> Int

    return SoftSource(isrc, Amp, Phi, waveform, p, icomp)
end


function source_init(data::DataSoftSource, field::Field3D, t)
    (; geometry, amplitude, phase, waveform, p, frequency, component) = data
    (; grid) = field
    (; Nx, Ny, Nz, x, y, z) = grid

    geom = [geometry(x[ix], y[iy], z[iz]) for ix=1:Nx, iy=1:Ny, iz=1:Nz]
    isrc = findall(geom)

    Amp, Phi = zeros(size(isrc)), zeros(size(isrc))
    for (i, ici) in enumerate(isrc)
        ix, iy, iz = ici[1], ici[2], ici[3]
        Amp[i] = amplitude(x[ix], y[iy], z[iz])
        Phi[i] = phase(x[ix], y[iy], z[iz]) / frequency
    end

    icomp = findfirst(isequal(component), fieldnames(typeof(field)))   # Symbol -> Int

    return SoftSource(isrc, Amp, Phi, waveform, p, icomp)
end


function add_source!(model, source::SoftSource, it)
    (; field, t) = model
    (; isrc, icomp, Amp, Phi, waveform, p) = source
    FC = getfield(field, icomp)
    @views @. FC[isrc] = FC[isrc] + Amp * waveform(t[it] - Phi, (p,))
end


# ******************************************************************************************
# Hard
# ******************************************************************************************
struct DataHardSource{FG, FA, FP, FW, P, T, S}
    geometry :: FG
    amplitude :: FA
    phase :: FP
    waveform :: FW
    p :: P
    frequency :: T
    component :: S
end


struct HardSource{C, A, F, P} <: Source
    isrc :: C
    Amp :: A
    Phi :: A
    waveform :: F
    p :: P
    icomp :: Int
end

@adapt_structure HardSource


function HardSource(; geometry, amplitude, phase, waveform, p, frequency, component)
    return DataHardSource(geometry, amplitude, phase, waveform, p, frequency, component)
end


function source_init(data::DataHardSource, field, t)
    (; geometry, amplitude, phase, waveform, p, frequency, component) = data
    sdata = DataSoftSource(geometry, amplitude, phase, waveform, p, frequency, component)
    ssource = source_init(sdata, field, t)
    (; isrc, Amp, Phi, waveform, p, icomp) = ssource
    return HardSource(isrc, Amp, Phi, waveform, p, icomp)
end


function add_source!(model, source::HardSource, it)
    (; field, t) = model
    (; isrc, icomp, Amp, Phi, waveform, p) = source
    FC = getfield(field, icomp)
    @views @. FC[isrc] = Amp * waveform(t[it] - Phi, (p,))
end


# ******************************************************************************************
# TFSF
# ******************************************************************************************
struct DataTFSFSource{S, P}
    fname :: S
    tfsf_box :: P
end


function TFSFSource(; fname, tfsf_box)
    return DataTFSFSource(fname, tfsf_box)
end


# ------------------------------------------------------------------------------------------
# TFSF 1D
# ------------------------------------------------------------------------------------------
struct TFSFSource1D{A} <: Source
    iz1 :: Int
    iz2 :: Int
    lincHy :: A
    rincHy :: A
    lincEx :: A
    rincEx :: A
end

@adapt_structure TFSFSource1D


function source_init(data::DataTFSFSource, field::Field1D, t)
    (; fname, tfsf_box) = data
    (; grid) = field
    (; z) = grid

    fp = HDF5.h5open(fname, "r")
    zex = HDF5.read(fp, "z")
    tex = HDF5.read(fp, "t")
    exHy = HDF5.read(fp, "Hy")
    exEx = HDF5.read(fp, "Ex")
    HDF5.close(fp)

    z1, z2 = tfsf_box
    iz1 = argmin(abs.(zex .- z1))
    iz2 = argmin(abs.(zex .- z2))

    # # left:
    # lincEx = exEx[iz1,:]
    # lincHy = exHy[iz1,:]

    # # right:
    # rincEx = exEx[iz2,:]
    # rincHy = exHy[iz2,:]

    itpHy = linear_interpolation((zex,tex), exHy; extrapolation_bc=Flat())
    itpEx = linear_interpolation((zex,tex), exEx; extrapolation_bc=Flat())

    # left:
    lincHy = @. itpHy(z[iz1], t)   # use z[iz1] instead of zl, since they can be different
    lincEx = @. itpEx(z[iz1], t)

    # right:
    rincHy = @. itpHy(z[iz2], t)   # use z[iz2] instead of zr, since they can be different
    rincEx = @. itpEx(z[iz2], t)

    return TFSFSource1D(iz1, iz2, lincHy, rincHy, lincEx, rincEx)
end


function add_source!(model, source::TFSFSource1D, it)
    (; field, dt) = model
    (; iz1, iz2, lincHy, rincHy, lincEx, rincEx) = source
    (; grid, Hy, Ex, Dx) = field
    (; dz) = grid

    Hy[iz1] += dt / (MU0*dz) * lincEx[it]
    Ex[iz1] += dt / (EPS0*dz) * lincHy[it]
    Dx[iz1] += dt / dz * lincHy[it]

    Hy[iz2] -= dt / (MU0*dz) * rincEx[it]
    Ex[iz2] -= dt / (EPS0*dz) * rincHy[it]
    Dx[iz2] -= dt / dz * rincHy[it]
    return nothing
end


# ------------------------------------------------------------------------------------------
# TFSF 2D
# ------------------------------------------------------------------------------------------
struct TFSFSource2D{A} <: Source
    ix1 :: Int
    ix2 :: Int
    iz1 :: Int
    iz2 :: Int
    lincHy :: A
    rincHy :: A
    bincHy :: A
    tincHy :: A
    bincEx :: A
    tincEx :: A
    lincEz :: A
    rincEz :: A
end

@adapt_structure TFSFSource2D


function source_init(data::DataTFSFSource, field::Field2D, t)
    (; fname, tfsf_box) = data
    (; grid) = field
    (; x, z) = grid

    fp = HDF5.h5open(fname, "r")
    xex = HDF5.read(fp, "z")
    zex = HDF5.read(fp, "z")
    tex = HDF5.read(fp, "t")
    exHy = HDF5.read(fp, "Hy")
    exEx = HDF5.read(fp, "Ex")
    exEz = HDF5.read(fp, "Ez")
    HDF5.close(fp)

    x1, x2, z1, z2 = tfsf_box
    ix1 = argmin(abs.(xex .- x1))
    ix2 = argmin(abs.(xex .- x2))
    iz1 = argmin(abs.(zex .- z1))
    iz2 = argmin(abs.(zex .- z2))

    # # left:
    # lincHy = exHy[ix1,iz1:iz2-1,:]
    # lincEz = exEz[ix1,iz1:iz2-1,:]

    # # right:
    # rincHy = exHy[ix2,iz1:iz2-1,:]
    # rincEz = exEz[ix2,iz1:iz2-1,:]

    # # bottom:
    # bincHy = exHy[ix1:ix2-1,iz1,:]
    # bincEx = exEx[ix1:ix2-1,iz1,:]

    # # top:
    # tincHy = exHy[ix1:ix2-1,iz2,:]
    # tincEx = exEx[ix1:ix2-1,iz2,:]

    itpHy = linear_interpolation((xex,zex,tex), exHy; extrapolation_bc=Flat())
    itpEx = linear_interpolation((xex,zex,tex), exEx; extrapolation_bc=Flat())
    itpEz = linear_interpolation((xex,zex,tex), exEz; extrapolation_bc=Flat())

    Nxi = ix2 - ix1
    Nzi = iz2 - iz1
    Nt = length(t)

    # left:
    lincHy, lincEz = zeros(Nzi,Nt), zeros(Nzi,Nt)
    for it=1:Nt, iz=iz1:iz2-1
        lincHy[iz-iz1+1,it] = itpHy(x[ix1], z[iz], t[it])
        lincEz[iz-iz1+1,it] = itpEz(x[ix1], z[iz], t[it])
    end

    # right:
    rincHy, rincEz = zeros(Nzi,Nt), zeros(Nzi,Nt)
    for it=1:Nt, iz=iz1:iz2-1
        rincHy[iz-iz1+1,it] = itpHy(x[ix2], z[iz], t[it])
        rincEz[iz-iz1+1,it] = itpEz(x[ix2], z[iz], t[it])
    end

    # bottom:
    bincHy, bincEx = zeros(Nxi,Nt), zeros(Nxi,Nt)
    for it=1:Nt, ix=ix1:ix2-1
        bincHy[ix-ix1+1,it] = itpHy(x[ix], z[iz1], t[it])
        bincEx[ix-ix1+1,it] = itpEx(x[ix], z[iz1], t[it])
    end

    # top:
    tincHy, tincEx = zeros(Nxi,Nt), zeros(Nxi,Nt)
    for it=1:Nt, ix=ix1:ix2-1
        tincHy[ix-ix1+1,it] = itpHy(x[ix], z[iz2], t[it])
        tincEx[ix-ix1+1,it] = itpEx(x[ix], z[iz2], t[it])
    end

    return TFSFSource2D(
        ix1, ix2, iz1, iz2, lincHy, rincHy, bincHy, tincHy, bincEx, tincEx, lincEz, rincEz,
    )
end


function add_source!(model, source::TFSFSource2D, it)
    (; field, dt) = model
    (; ix1, ix2, iz1, iz2) = source
    (; lincHy, rincHy, bincHy, tincHy, bincEx, tincEx, lincEz, rincEz) = source
    (; grid, Hy, Dx, Dz, Ex, Ez) = field
    (; dx, dz) = grid

    # left:
    @views @. Hy[ix1,iz1:iz2-1] -= dt / (MU0*dx) * lincEz[:,it]
    @views @. Ez[ix1,iz1:iz2-1] -= dt / (EPS0*dx) * lincHy[:,it]
    @views @. Dz[ix1,iz1:iz2-1] -= dt / dx * lincHy[:,it]

    # right:
    @views @. Hy[ix2,iz1:iz2-1] += dt / (MU0*dx) * rincEz[:,it]
    @views @. Ez[ix2,iz1:iz2-1] += dt / (EPS0*dx) * rincHy[:,it]
    @views @. Dz[ix2,iz1:iz2-1] += dt / dx * rincHy[:,it]

    # bottom:
    @views @. Hy[ix1:ix2-1,iz1] += dt / (MU0*dz) * bincEx[:,it]
    @views @. Ex[ix1:ix2-1,iz1] += dt / (EPS0*dz) * bincHy[:,it]
    @views @. Dx[ix1:ix2-1,iz1] += dt / dz * bincHy[:,it]

    # top:
    @views @. Hy[ix1:ix2-1,iz2] -= dt / (MU0*dz) * tincEx[:,it]
    @views @. Ex[ix1:ix2-1,iz2] -= dt / (EPS0*dz) * tincHy[:,it]
    @views @. Dx[ix1:ix2-1,iz2] -= dt / dz * tincHy[:,it]
    return nothing
end
