abstract type Source end


# ******************************************************************************************
# Soft
# ******************************************************************************************
struct SoftSource{C, S, A, F, P} <: Source
    isrc :: C
    component :: S
    Amp :: A
    Phi :: A
    waveform :: F
    p :: P
end

@adapt_structure SoftSource


function SoftSource(
    grid::Grid1D; geometry, component, frequency, amplitude, phase, waveform, p,
)
    (; z) = grid

    geom = @. geometry(z)
    isrc = findall(geom)

    Amp, Phi = zeros(size(isrc)), zeros(size(isrc))
    for (i, iz) in enumerate(isrc)
        Amp[i] = amplitude(z[iz])
        Phi[i] = phase(z[iz]) / frequency
    end

    return SoftSource(isrc, component, Amp, Phi, waveform, p)
end


function SoftSource(
    grid::Grid2D; geometry, component, frequency, amplitude, phase, waveform, p,
)
    (; Nx, Nz, x, z) = grid

    geom = [geometry(x[ix], z[iz]) for ix=1:Nx, iz=1:Nz]
    isrc = findall(geom)

    Amp, Phi = zeros(size(isrc)), zeros(size(isrc))
    for (i, ici) in enumerate(isrc)
        ix, iz = ici[1], ici[2]
        Amp[i] = amplitude(x[ix], z[iz])
        Phi[i] = phase(x[ix], z[iz]) / frequency
    end

    return SoftSource(isrc, component, Amp, Phi, waveform, p)
end


function SoftSource(
    grid::Grid3D; geometry, component, frequency, amplitude, phase, waveform, p,
)
    (; Nx, Ny, Nz, x, y, z) = grid

    geom = [geometry(x[ix], y[iy], z[iz]) for ix=1:Nx, iy=1:Ny, iz=1:Nz]
    isrc = findall(geom)

    Amp, Phi = zeros(size(isrc)), zeros(size(isrc))
    for (i, ici) in enumerate(isrc)
        ix, iy, iz = ici[1], ici[2], ici[3]
        Amp[i] = amplitude(x[ix], y[iy], z[iz])
        Phi[i] = phase(x[ix], y[iy], z[iz]) / frequency
    end

    return SoftSource(isrc, component, Amp, Phi, waveform, p)
end


function add_source!(model, source::SoftSource, it)
    (; field, t) = model
    (; isrc, component, Amp, Phi, waveform, p) = source
    FC = getfield(field, component)
    @views @. FC[isrc] = FC[isrc] + Amp * waveform(t[it] - Phi, (p,))
end


# ******************************************************************************************
# Hard
# ******************************************************************************************
struct HardSource{C, S, A, F, P} <: Source
    isrc :: C
    component :: S
    Amp :: A
    Phi :: A
    waveform :: F
    p :: P
end

@adapt_structure HardSource


function HardSource(grid; geometry, component, frequency, amplitude, phase, waveform, p)
    source = SoftSource(grid; geometry, component, frequency, amplitude, phase, waveform, p)
    (; isrc, component, Amp, Phi, waveform, p) = source
    return HardSource(isrc, component, Amp, Phi, waveform, p)
end


function add_source!(model, source::HardSource, it)
    (; field, t) = model
    (; isrc, component, Amp, Phi, waveform, p) = source
    FC = getfield(field, component)
    @views @. FC[isrc] = Amp * waveform(t[it] - Phi, (p,))
end


# ******************************************************************************************
# TFSF 1D
# ******************************************************************************************
struct TFSFSource1D{A} <: Source
    iz1 :: Int
    iz2 :: Int
    lincHy :: A
    rincHy :: A
    lincEx :: A
    rincEx :: A
end

@adapt_structure TFSFSource1D


function TFSFSource(grid::Grid1D; fname, tfsf_box)
    fp = HDF5.h5open(fname, "r")
    zex = HDF5.read(fp, "z")
    tex = HDF5.read(fp, "t")
    exHy = HDF5.read(fp, "Hy")
    exEx = HDF5.read(fp, "Ex")
    HDF5.close(fp)

    z1, z2 = tfsf_box
    iz1 = argmin(abs.(zex .- z1))
    iz2 = argmin(abs.(zex .- z2))

    # left:
    lincEx = exEx[iz1,:]
    lincHy = exHy[iz1,:]

    # right:
    rincEx = exEx[iz2,:]
    rincHy = exHy[iz2,:]

    # itpHy = linear_interpolation((zex,tex), exHy; extrapolation_bc=Flat())
    # itpEx = linear_interpolation((zex,tex), exEx; extrapolation_bc=Flat())

    # # left:
    # lincHy = @. itpHy(z[iz1], t)   # use z[iz1] instead of zl, since they can be different
    # lincEx = @. itpEx(z[iz1], t)

    # # right:
    # rincHy = @. itpHy(z[iz2], t)   # use z[iz2] instead of zr, since they can be different
    # rincEx = @. itpEx(z[iz2], t)

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


# ******************************************************************************************
# TFSF 2D
# ******************************************************************************************
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


function TFSFSource(grid::Grid2D; fname, tfsf_box)
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

    # left:
    lincHy = exHy[ix1,iz1:iz2-1,:]
    lincEz = exEz[ix1,iz1:iz2-1,:]

    # right:
    rincHy = exHy[ix2,iz1:iz2-1,:]
    rincEz = exEz[ix2,iz1:iz2-1,:]

    # bottom:
    bincHy = exHy[ix1:ix2-1,iz1,:]
    bincEx = exEx[ix1:ix2-1,iz1,:]

    # top:
    tincHy = exHy[ix1:ix2-1,iz2,:]
    tincEx = exEx[ix1:ix2-1,iz2,:]

    # itpHy = linear_interpolation((xex,zex,tex), exHy; extrapolation_bc=Flat())
    # itpEx = linear_interpolation((xex,zex,tex), exEx; extrapolation_bc=Flat())
    # itpEz = linear_interpolation((xex,zex,tex), exEz; extrapolation_bc=Flat())

    # Nxi = ix2 - ix1
    # Nzi = iz2 - iz1

    # # left:
    # lincHy, lincEz = zeros(Nzi,Nt), zeros(Nzi,Nt)
    # for it=1:Nt, iz=iz1:iz2-1
    #     lincHy[iz-iz1+1,it] = itpHy(x[ix1], z[iz], t[it])
    #     lincEz[iz-iz1+1,it] = itpEz(x[ix1], z[iz], t[it])
    # end

    # # right:
    # rincHy, rincEz = zeros(Nzi,Nt), zeros(Nzi,Nt)
    # for it=1:Nt, iz=iz1:iz2-1
    #     rincHy[iz-iz1+1,it] = itpHy(x[ix2], z[iz], t[it])
    #     rincEz[iz-iz1+1,it] = itpEz(x[ix2], z[iz], t[it])
    # end

    # # bottom:
    # bincHy, bincEx = zeros(Nxi,Nt), zeros(Nxi,Nt)
    # for it=1:Nt, ix=ix1:ix2-1
    #     bincHy[ix-ix1+1,it] = itpHy(x[ix], z[iz1], t[it])
    #     bincEx[ix-ix1+1,it] = itpEx(x[ix], z[iz1], t[it])
    # end

    # # top:
    # tincHy, tincEx = zeros(Nxi,Nt), zeros(Nxi,Nt)
    # for it=1:Nt, ix=ix1:ix2-1
    #     tincHy[ix-ix1+1,it] = itpHy(x[ix], z[iz2], t[it])
    #     tincEx[ix-ix1+1,it] = itpEx(x[ix], z[iz2], t[it])
    # end

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


# ******************************************************************************************
# Util
# ******************************************************************************************
"""
Converts the type of the 'component' field from 'Symbol' to 'Int'
"""
function isbitify(source::Union{SoftSource,HardSource}, field)
    (; isrc, component, Amp, Phi, waveform, p) = source
    if typeof(component) == Symbol
        icomp = findfirst(isequal(component), fieldnames(typeof(field)))
    else
        icomp = component
    end
    if typeof(source) <: SoftSource
        return SoftSource(isrc, icomp, Amp, Phi, waveform, p)
    elseif typeof(source) <: HardSource
        return HardSource(isrc, icomp, Amp, Phi, waveform, p)
    end
end


function isbitify(source::Union{TFSFSource1D,TFSFSource2D}, field)
    return source
end
