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


function SoftSource(;
    geometry, amplitude=nothing, phase=nothing, frequency=nothing, waveform, p, component,
)
    if !isnothing(phase) && isnothing(frequency)
        error(
            "I need to know the source frequency in order to convert its phase into " *
            "a proper time delay"
        )
    end
    return DataSoftSource(geometry, amplitude, phase, waveform, p, frequency, component)
end


function source_init(data::DataSoftSource, field::Field1D, t)
    (; geometry, amplitude, phase, waveform, p, frequency, component) = data
    (; grid) = field
    (; z) = grid

    if isnothing(amplitude)
        amplitude = z -> 1
    end
    if isnothing(phase)
        phase = z -> 0
        frequency = 1
    end

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

    if isnothing(amplitude)
        amplitude = (x,z) -> 1
    end
    if isnothing(phase)
        phase = (x,z) -> 0
        frequency = 1
    end

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

    if isnothing(amplitude)
        amplitude = (x,y,z) -> 1
    end
    if isnothing(phase)
        phase = (x,y,z) -> 0
        frequency = 1
    end

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


function HardSource(;
    geometry, amplitude=nothing, phase=nothing, frequency=nothing, waveform, p, component,
)
    if !isnothing(phase) && isnothing(frequency)
        error(
            "I need to know the source frequency in order to convert its phase into " *
            "a proper time delay"
        )
    end
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


function TFSFSource(; fname)
    return DataTFSFSource(fname, nothing)
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
    (; fname) = data
    (; grid) = field
    (; z) = grid

    fp = HDF5.h5open(fname, "r")
    tfsf_box = HDF5.read(fp, "tfsf_box")
    tex = HDF5.read(fp, "t")
    lincHy_ex = HDF5.read(fp, "lincHy")   # left
    lincEx_ex = HDF5.read(fp, "lincEx")
    rincHy_ex = HDF5.read(fp, "rincHy")   # right
    rincEx_ex = HDF5.read(fp, "rincEx")
    # temporarily for debug:
    # zex = HDF5.read(fp, "z")
    # Hy_ex = HDF5.read(fp, "Hy")
    # Ex_ex = HDF5.read(fp, "Ex")
    HDF5.close(fp)

    z1, z2 = tfsf_box
    iz1 = argmin(abs.(z .- z1))
    iz2 = argmin(abs.(z .- z2))

    itp(x,y) = linear_interpolation(x, y; extrapolation_bc=Flat())

    # ......................................................................................
    # left:
    itp_lincHy = itp(tex, lincHy_ex)
    itp_lincEx = itp(tex, lincEx_ex)
    lincHy = @. itp_lincHy(t)
    lincEx = @. itp_lincEx(t)

    # right:
    itp_rincHy = itp(tex, rincHy_ex)
    itp_rincEx = itp(tex, rincEx_ex)
    rincHy = @. itp_rincHy(t)
    rincEx = @. itp_rincEx(t)

    # ......................................................................................
    # # left:
    # lincHy = Hy_ex[iz1,:]
    # lincEx = Ex_ex[iz1,:]

    # # right:
    # rincHy = Hy_ex[iz2,:]
    # rincEx = Ex_ex[iz2,:]

    # ......................................................................................
    # itpHy = itp((zex,tex), Hy_ex)
    # itpEx = itp((zex,tex), Ex_ex)

    # # left:
    # lincHy = @. itpHy(z[iz1], t)
    # lincEx = @. itpEx(z[iz1], t)

    # # right:
    # rincHy = @. itpHy(z[iz2], t)
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

    # (; Mh, Me) = model
    # Hy[iz1] += Mh[iz1]/dz * lincEx[it]
    # Ex[iz1] += Me[iz1]*dt/dz * lincHy[it]
    # Dx[iz1] += EPS0 * Me[iz1]*dt/dz * lincHy[it]

    # Hy[iz2] -= Mh[iz2]/dz * rincEx[it]
    # Ex[iz2] -= Me[iz2]*dt/dz * rincHy[it]
    # Dx[iz2] -= EPS0 * Me[iz1]*dt/dz * rincHy[it]

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
    (; fname) = data
    (; grid) = field
    (; x, z) = grid

    fp = HDF5.h5open(fname, "r")
    tfsf_box = HDF5.read(fp, "tfsf_box")
    xex = HDF5.read(fp, "x")
    zex = HDF5.read(fp, "z")
    tex = HDF5.read(fp, "t")
    lincHy_ex = HDF5.read(fp, "lincHy")   # left
    lincEz_ex = HDF5.read(fp, "lincEz")
    rincHy_ex = HDF5.read(fp, "rincHy")   # right
    rincEz_ex = HDF5.read(fp, "rincEz")
    bincHy_ex = HDF5.read(fp, "bincHy")   # bottom
    bincEx_ex = HDF5.read(fp, "bincEx")
    tincHy_ex = HDF5.read(fp, "tincHy")   # top
    tincEx_ex = HDF5.read(fp, "tincEx")
    HDF5.close(fp)

    x1, x2, z1, z2 = tfsf_box
    ix1 = argmin(abs.(x .- x1))
    ix2 = argmin(abs.(x .- x2))
    iz1 = argmin(abs.(z .- z1))
    iz2 = argmin(abs.(z .- z2))

    Nxi = ix2 - ix1
    Nzi = iz2 - iz1
    Nt = length(t)

    itp(x,y) = linear_interpolation(x, y; extrapolation_bc=Flat())

    # left:
    itp_lincHy = itp((zex,tex), lincHy_ex)
    itp_lincEz = itp((zex,tex), lincEz_ex)
    lincHy, lincEz = zeros(Nzi,Nt), zeros(Nzi,Nt)
    for it=1:Nt, iz=iz1:iz2-1
        lincHy[iz-iz1+1,it] = itp_lincHy(z[iz], t[it])
        lincEz[iz-iz1+1,it] = itp_lincEz(z[iz], t[it])
    end

    # right:
    itp_rincHy = itp((zex,tex), rincHy_ex)
    itp_rincEz = itp((zex,tex), rincEz_ex)
    rincHy, rincEz = zeros(Nzi,Nt), zeros(Nzi,Nt)
    for it=1:Nt, iz=iz1:iz2-1
        rincHy[iz-iz1+1,it] = itp_rincHy(z[iz], t[it])
        rincEz[iz-iz1+1,it] = itp_rincEz(z[iz], t[it])
    end

    # bottom:
    itp_bincHy = itp((xex,tex), bincHy_ex)
    itp_bincEx = itp((xex,tex), bincEx_ex)
    bincHy, bincEx = zeros(Nxi,Nt), zeros(Nxi,Nt)
    for it=1:Nt, ix=ix1:ix2-1
        bincHy[ix-ix1+1,it] = itp_bincHy(x[ix], t[it])
        bincEx[ix-ix1+1,it] = itp_bincEx(x[ix], t[it])
    end

    # top:
    itp_tincHy = itp((xex,tex), tincHy_ex)
    itp_tincEx = itp((xex,tex), tincEx_ex)
    tincHy, tincEx = zeros(Nxi,Nt), zeros(Nxi,Nt)
    for it=1:Nt, ix=ix1:ix2-1
        tincHy[ix-ix1+1,it] = itp_tincHy(x[ix], t[it])
        tincEx[ix-ix1+1,it] = itp_tincEx(x[ix], t[it])
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


# ------------------------------------------------------------------------------------------
# TFSF 3D
# ------------------------------------------------------------------------------------------
struct TFSFSource3D{A} <: Source
    ix1 :: Int
    ix2 :: Int
    iy1 :: Int
    iy2 :: Int
    iz1 :: Int
    iz2 :: Int
    # 1yz:
    incEz_1yz :: A
    incEy_1yz :: A
    incHz_1yz :: A
    incHy_1yz :: A
    # 2yz:
    incEz_2yz :: A
    incEy_2yz :: A
    incHz_2yz :: A
    incHy_2yz :: A
    # x1z:
    incEz_x1z :: A
    incEx_x1z :: A
    incHz_x1z :: A
    incHx_x1z :: A
    # x2z:
    incEz_x2z :: A
    incEx_x2z :: A
    incHz_x2z :: A
    incHx_x2z :: A
    # xy1:
    incEy_xy1 :: A
    incEx_xy1 :: A
    incHy_xy1 :: A
    incHx_xy1 :: A
    # xy2:
    incEy_xy2 :: A
    incEx_xy2 :: A
    incHy_xy2 :: A
    incHx_xy2 :: A
end

@adapt_structure TFSFSource3D


function source_init(data::DataTFSFSource, field::Field3D, t)
    (; fname) = data
    (; grid) = field
    (; x, y, z) = grid

    fp = HDF5.h5open(fname, "r")
    tfsf_box = HDF5.read(fp, "tfsf_box")
    xex = HDF5.read(fp, "x")
    zex = HDF5.read(fp, "z")
    tex = HDF5.read(fp, "t")
    # 1yz:
    incEz_1yz = HDF5.read(fp, "incEz_1yz")
    incEy_1yz = HDF5.read(fp, "incEy_1yz")
    incHz_1yz = HDF5.read(fp, "incHz_1yz")
    incHy_1yz = HDF5.read(fp, "incHy_1yz")
    # 2yz:
    incEz_2yz = HDF5.read(fp, "incEz_2yz")
    incEy_2yz = HDF5.read(fp, "incEy_2yz")
    incHz_2yz = HDF5.read(fp, "incHz_2yz")
    incHy_2yz = HDF5.read(fp, "incHy_2yz")
    # x1z:
    incEz_x1z = HDF5.read(fp, "incEz_x1z")
    incEx_x1z = HDF5.read(fp, "incEx_x1z")
    incHz_x1z = HDF5.read(fp, "incHz_x1z")
    incHx_x1z = HDF5.read(fp, "incHx_x1z")
    # x2z:
    incEz_x2z = HDF5.read(fp, "incEz_x2z")
    incEx_x2z = HDF5.read(fp, "incEx_x2z")
    incHz_x2z = HDF5.read(fp, "incHz_x2z")
    incHx_x2z = HDF5.read(fp, "incHx_x2z")
    # xy1:
    incEy_xy1 = HDF5.read(fp, "incEy_xy1")
    incEx_xy1 = HDF5.read(fp, "incEx_xy1")
    incHy_xy1 = HDF5.read(fp, "incHy_xy1")
    incHx_xy1 = HDF5.read(fp, "incHx_xy1")
    # xy2:
    incEy_xy2 = HDF5.read(fp, "incEy_xy2")
    incEx_xy2 = HDF5.read(fp, "incEx_xy2")
    incHy_xy2 = HDF5.read(fp, "incHy_xy2")
    incHx_xy2 = HDF5.read(fp, "incHx_xy2")
    HDF5.close(fp)

    x1, x2, y1, y2, z1, z2 = tfsf_box
    ix1 = argmin(abs.(x .- x1))
    ix2 = argmin(abs.(x .- x2))
    iy1 = argmin(abs.(y .- y1))
    iy2 = argmin(abs.(y .- y2))
    iz1 = argmin(abs.(z .- z1))
    iz2 = argmin(abs.(z .- z2))

    Nxi = ix2 - ix1
    Nyi = iy2 - iy1
    Nzi = iz2 - iz1
    Nt = length(t)

    return TFSFSource3D(
        ix1, ix2, iy1, iy2, iz1, iz2,
        incEz_1yz, incEy_1yz, incHz_1yz, incHy_1yz,
        incEz_2yz, incEy_2yz, incHz_2yz, incHy_2yz,
        incEz_x1z, incEx_x1z, incHz_x1z, incHx_x1z,
        incEz_x2z, incEx_x2z, incHz_x2z, incHx_x2z,
        incEy_xy1, incEx_xy1, incHy_xy1, incHx_xy1,
        incEy_xy2, incEx_xy2, incHy_xy2, incHx_xy2,
    )
end


function add_source!(model, source::TFSFSource3D, it)
    (; field, dt) = model
    (; ix1, ix2, iy1, iy2, iz1, iz2,
       incEz_1yz, incEy_1yz, incHz_1yz, incHy_1yz,
       incEz_2yz, incEy_2yz, incHz_2yz, incHy_2yz,
       incEz_x1z, incEx_x1z, incHz_x1z, incHx_x1z,
       incEz_x2z, incEx_x2z, incHz_x2z, incHx_x2z,
       incEy_xy1, incEx_xy1, incHy_xy1, incHx_xy1,
       incEy_xy2, incEx_xy2, incHy_xy2, incHx_xy2) = source
    (; grid, Hx, Hy, Hz, Dx, Dy, Dz, Ex, Ey, Ez) = field
    (; dx, dy, dz) = grid

    # 1yz:
    @views @. Hy[ix1,iy1:iy2-1,iz1:iz2-1] -= dt / (MU0*dx) * incEz_1yz[:,:,it]
    @views @. Hz[ix1,iy1:iy2-1,iz1:iz2-1] += dt / (MU0*dx) * incEy_1yz[:,:,it]
    @views @. Ey[ix1,iy1:iy2-1,iz1:iz2-1] += dt / (EPS0*dx) * incHz_1yz[:,:,it]
    @views @. Ez[ix1,iy1:iy2-1,iz1:iz2-1] -= dt / (EPS0*dx) * incHy_1yz[:,:,it]
    @views @. Dy[ix1,iy1:iy2-1,iz1:iz2-1] += dt / dx * incHz_1yz[:,:,it]
    @views @. Dz[ix1,iy1:iy2-1,iz1:iz2-1] -= dt / dx * incHy_1yz[:,:,it]

    # 2yz:
    @views @. Hy[ix2,iy1:iy2-1,iz1:iz2-1] += dt / (MU0*dx) * incEz_2yz[:,:,it]
    @views @. Hz[ix2,iy1:iy2-1,iz1:iz2-1] -= dt / (MU0*dx) * incEy_2yz[:,:,it]
    @views @. Ey[ix2,iy1:iy2-1,iz1:iz2-1] -= dt / (EPS0*dx) * incHz_2yz[:,:,it]
    @views @. Ez[ix2,iy1:iy2-1,iz1:iz2-1] += dt / (EPS0*dx) * incHy_2yz[:,:,it]
    @views @. Dy[ix2,iy1:iy2-1,iz1:iz2-1] -= dt / dx * incHz_2yz[:,:,it]
    @views @. Dz[ix2,iy1:iy2-1,iz1:iz2-1] += dt / dx * incHy_2yz[:,:,it]

    # x1z:
    @views @. Hx[ix1:ix2-1,iy1,iz1:iz2-1] += dt / (MU0*dy) * incEz_x1z[:,:,it]
    @views @. Hz[ix1:ix2-1,iy1,iz1:iz2-1] -= dt / (MU0*dy) * incEx_x1z[:,:,it]
    @views @. Ex[ix1:ix2-1,iy1,iz1:iz2-1] -= dt / (EPS0*dy) * incHz_x1z[:,:,it]
    @views @. Ez[ix1:ix2-1,iy1,iz1:iz2-1] += dt / (EPS0*dy) * incHx_x1z[:,:,it]
    @views @. Dx[ix1:ix2-1,iy1,iz1:iz2-1] -= dt / dy * incHz_x1z[:,:,it]
    @views @. Dz[ix1:ix2-1,iy1,iz1:iz2-1] += dt / dy * incHx_x1z[:,:,it]

    # x2z:
    @views @. Hx[ix1:ix2-1,iy2,iz1:iz2-1] -= dt / (MU0*dy) * incEz_x2z[:,:,it]
    @views @. Hz[ix1:ix2-1,iy2,iz1:iz2-1] += dt / (MU0*dy) * incEx_x2z[:,:,it]
    @views @. Ex[ix1:ix2-1,iy2,iz1:iz2-1] += dt / (EPS0*dy) * incHz_x2z[:,:,it]
    @views @. Ez[ix1:ix2-1,iy2,iz1:iz2-1] -= dt / (EPS0*dy) * incHx_x2z[:,:,it]
    @views @. Dx[ix1:ix2-1,iy2,iz1:iz2-1] += dt / dy * incHz_x2z[:,:,it]
    @views @. Dz[ix1:ix2-1,iy2,iz1:iz2-1] -= dt / dy * incHx_x2z[:,:,it]

    # xy1:
    @views @. Hx[ix1:ix2-1,iy1:iy2-1,iz1] -= dt / (MU0*dz) * incEy_xy1[:,:,it]
    @views @. Hy[ix1:ix2-1,iy1:iy2-1,iz1] += dt / (MU0*dz) * incEx_xy1[:,:,it]
    @views @. Ex[ix1:ix2-1,iy1:iy2-1,iz1] += dt / (EPS0*dz) * incHy_xy1[:,:,it]
    @views @. Ey[ix1:ix2-1,iy1:iy2-1,iz1] -= dt / (EPS0*dz) * incHx_xy1[:,:,it]
    @views @. Dx[ix1:ix2-1,iy1:iy2-1,iz1] += dt / dz * incHy_xy1[:,:,it]
    @views @. Dy[ix1:ix2-1,iy1:iy2-1,iz1] -= dt / dz * incHx_xy1[:,:,it]

    # xy2
    @views @. Hx[ix1:ix2-1,iy1:iy2-1,iz2] += dt / (MU0*dz) * incEy_xy2[:,:,it]
    @views @. Hy[ix1:ix2-1,iy1:iy2-1,iz2] -= dt / (MU0*dz) * incEx_xy2[:,:,it]
    @views @. Ex[ix1:ix2-1,iy1:iy2-1,iz2] -= dt / (EPS0*dz) * incHy_xy2[:,:,it]
    @views @. Ey[ix1:ix2-1,iy1:iy2-1,iz2] += dt / (EPS0*dz) * incHx_xy2[:,:,it]
    @views @. Dx[ix1:ix2-1,iy1:iy2-1,iz2] -= dt / dz * incHy_xy2[:,:,it]
    @views @. Dy[ix1:ix2-1,iy1:iy2-1,iz2] += dt / dz * incHx_xy2[:,:,it]

    return nothing
end


# ******************************************************************************************
# TFSF record
# ******************************************************************************************
# ------------------------------------------------------------------------------------------
# TFSF record 1D
# ------------------------------------------------------------------------------------------
function prepare_tfsf_record(model::Model1D, tfsf_box, tfsf_fname)
    (; field, Nt, t) = model
    (; grid, Hy) = field
    (; Nz, z) = grid

    z1, z2 = tfsf_box
    iz1 = argmin(abs.(z .- z1))
    iz2 = argmin(abs.(z .- z2))

    T = eltype(Hy)
    HDF5.h5open(tfsf_fname, "w") do fp
        fp["t"] = collect(t)
        fp["tfsf_box"] = collect(tfsf_box)
        HDF5.create_dataset(fp, "lincHy", T, Nt)   # left
        HDF5.create_dataset(fp, "lincEx", T, Nt)
        HDF5.create_dataset(fp, "rincHy", T, Nt)   # right
        HDF5.create_dataset(fp, "rincEx", T, Nt)
        # temporarily for debug:
        fp["z"] = collect(z)
        HDF5.create_dataset(fp, "Hy", T, (Nz,Nt))
        HDF5.create_dataset(fp, "Ex", T, (Nz,Nt))
    end

    return DataTFSFSource(tfsf_fname, (iz1, iz2))
end


function write_tfsf_record(model::Model1D, tfsf_data, it)
    (; field) = model
    (; Hy, Ex) = field
    (; fname, tfsf_box) = tfsf_data
    iz1, iz2 = tfsf_box
    HDF5.h5open(fname, "r+") do fp
        fp["lincHy"][it] = collect(Hy[iz1])   # left
        fp["lincEx"][it] = collect(Ex[iz1])
        fp["rincHy"][it] = collect(Hy[iz2])   # right
        fp["rincEx"][it] = collect(Ex[iz2])
        # temporarily for debug:
        fp["Hy"][:,it] = collect(Hy)   # test
        fp["Ex"][:,it] = collect(Ex)   # test
    end
    return nothing
end


# ------------------------------------------------------------------------------------------
# TFSF record 2D
# ------------------------------------------------------------------------------------------
function prepare_tfsf_record(model::Model2D, tfsf_box, tfsf_fname)
    (; field, Nt, t) = model
    (; grid, Hy) = field
    (; x, z) = grid

    x1, x2, z1, z2 = tfsf_box
    ix1 = argmin(abs.(x .- x1))
    ix2 = argmin(abs.(x .- x2))
    iz1 = argmin(abs.(z .- z1))
    iz2 = argmin(abs.(z .- z2))

    Nxi = ix2 - ix1
    Nzi = iz2 - iz1

    T = eltype(Hy)
    HDF5.h5open(tfsf_fname, "w") do fp
        fp["x"] = collect(x[ix1:ix2-1])
        fp["z"] = collect(z[iz1:iz2-1])
        fp["t"] = collect(t)
        fp["tfsf_box"] = collect(tfsf_box)
        HDF5.create_dataset(fp, "lincHy", T, (Nzi, Nt))   # left
        HDF5.create_dataset(fp, "lincEz", T, (Nzi, Nt))
        HDF5.create_dataset(fp, "rincHy", T, (Nzi, Nt))   # right
        HDF5.create_dataset(fp, "rincEz", T, (Nzi, Nt))
        HDF5.create_dataset(fp, "bincHy", T, (Nxi, Nt))   # bottom
        HDF5.create_dataset(fp, "bincEx", T, (Nxi, Nt))
        HDF5.create_dataset(fp, "tincHy", T, (Nxi, Nt))   # top
        HDF5.create_dataset(fp, "tincEx", T, (Nxi, Nt))
    end

    return DataTFSFSource(tfsf_fname, (ix1, ix2, iz1, iz2))
end


function write_tfsf_record(model::Model2D, tfsf_data, it)
    (; field) = model
    (; Hy, Ex, Ez) = field
    (; fname, tfsf_box) = tfsf_data
    ix1, ix2, iz1, iz2 = tfsf_box
    HDF5.h5open(fname, "r+") do fp
        fp["lincHy"][:,it] = collect(Hy[ix1,iz1:iz2-1])   # left
        fp["lincEz"][:,it] = collect(Ez[ix1,iz1:iz2-1])
        fp["rincHy"][:,it] = collect(Hy[ix2,iz1:iz2-1])   # right
        fp["rincEz"][:,it] = collect(Ez[ix2,iz1:iz2-1])
        fp["bincHy"][:,it] = collect(Hy[ix1:ix2-1,iz1])   # bottom
        fp["bincEx"][:,it] = collect(Ex[ix1:ix2-1,iz1])
        fp["tincHy"][:,it] = collect(Hy[ix1:ix2-1,iz2])   # top
        fp["tincEx"][:,it] = collect(Ex[ix1:ix2-1,iz2])
    end
    return nothing
end


# ------------------------------------------------------------------------------------------
# TFSF record 3D
# ------------------------------------------------------------------------------------------
function prepare_tfsf_record(model::Model3D, tfsf_box, tfsf_fname)
    (; field, Nt, t) = model
    (; grid, Hy) = field
    (; x, y, z) = grid

    x1, x2, y1, y2, z1, z2 = tfsf_box
    ix1 = argmin(abs.(x .- x1))
    ix2 = argmin(abs.(x .- x2))
    iy1 = argmin(abs.(y .- y1))
    iy2 = argmin(abs.(y .- y2))
    iz1 = argmin(abs.(z .- z1))
    iz2 = argmin(abs.(z .- z2))

    Nxi = ix2 - ix1
    Nyi = iy2 - iy1
    Nzi = iz2 - iz1

    T = eltype(Hy)
    HDF5.h5open(tfsf_fname, "w") do fp
        fp["x"] = collect(x[ix1:ix2-1])
        fp["y"] = collect(y[iy1:iy2-1])
        fp["z"] = collect(z[iz1:iz2-1])
        fp["t"] = collect(t)
        fp["tfsf_box"] = collect(tfsf_box)
        # 1yz:
        HDF5.create_dataset(fp, "incEz_1yz", T, (Nyi, Nzi, Nt))
        HDF5.create_dataset(fp, "incEy_1yz", T, (Nyi, Nzi, Nt))
        HDF5.create_dataset(fp, "incHz_1yz", T, (Nyi, Nzi, Nt))
        HDF5.create_dataset(fp, "incHy_1yz", T, (Nyi, Nzi, Nt))
        # 2yz:
        HDF5.create_dataset(fp, "incEz_2yz", T, (Nyi, Nzi, Nt))
        HDF5.create_dataset(fp, "incEy_2yz", T, (Nyi, Nzi, Nt))
        HDF5.create_dataset(fp, "incHz_2yz", T, (Nyi, Nzi, Nt))
        HDF5.create_dataset(fp, "incHy_2yz", T, (Nyi, Nzi, Nt))
        # x1z:
        HDF5.create_dataset(fp, "incEz_x1z", T, (Nxi, Nzi, Nt))
        HDF5.create_dataset(fp, "incEx_x1z", T, (Nxi, Nzi, Nt))
        HDF5.create_dataset(fp, "incHz_x1z", T, (Nxi, Nzi, Nt))
        HDF5.create_dataset(fp, "incHx_x1z", T, (Nxi, Nzi, Nt))
        # x2z:
        HDF5.create_dataset(fp, "incEz_x2z", T, (Nxi, Nzi, Nt))
        HDF5.create_dataset(fp, "incEx_x2z", T, (Nxi, Nzi, Nt))
        HDF5.create_dataset(fp, "incHz_x2z", T, (Nxi, Nzi, Nt))
        HDF5.create_dataset(fp, "incHx_x2z", T, (Nxi, Nzi, Nt))
        # xy1:
        HDF5.create_dataset(fp, "incEy_xy1", T, (Nxi, Nyi, Nt))
        HDF5.create_dataset(fp, "incEx_xy1", T, (Nxi, Nyi, Nt))
        HDF5.create_dataset(fp, "incHy_xy1", T, (Nxi, Nyi, Nt))
        HDF5.create_dataset(fp, "incHx_xy1", T, (Nxi, Nyi, Nt))
        # xy2:
        HDF5.create_dataset(fp, "incEy_xy2", T, (Nxi, Nyi, Nt))
        HDF5.create_dataset(fp, "incEx_xy2", T, (Nxi, Nyi, Nt))
        HDF5.create_dataset(fp, "incHy_xy2", T, (Nxi, Nyi, Nt))
        HDF5.create_dataset(fp, "incHx_xy2", T, (Nxi, Nyi, Nt))
    end

    return DataTFSFSource(tfsf_fname, (ix1, ix2, iy1, iy2, iz1, iz2))
end


function write_tfsf_record(model::Model3D, tfsf_data, it)
    (; field) = model
    (; Hx, Hy, Hz, Ex, Ey, Ez) = field
    (; fname, tfsf_box) = tfsf_data
    ix1, ix2, iy1, iy2, iz1, iz2 = tfsf_box
    HDF5.h5open(fname, "r+") do fp
        # 1yz:
        fp["incEz_1yz"][:,:,it] = collect(Ez[ix1,iy1:iy2-1,iz1:iz2-1])
        fp["incEy_1yz"][:,:,it] = collect(Ey[ix1,iy1:iy2-1,iz1:iz2-1])
        fp["incHz_1yz"][:,:,it] = collect(Hz[ix1,iy1:iy2-1,iz1:iz2-1])
        fp["incHy_1yz"][:,:,it] = collect(Hy[ix1,iy1:iy2-1,iz1:iz2-1])
        # 2yz:
        fp["incEz_2yz"][:,:,it] = collect(Ez[ix2,iy1:iy2-1,iz1:iz2-1])
        fp["incEy_2yz"][:,:,it] = collect(Ey[ix2,iy1:iy2-1,iz1:iz2-1])
        fp["incHz_2yz"][:,:,it] = collect(Hz[ix2,iy1:iy2-1,iz1:iz2-1])
        fp["incHy_2yz"][:,:,it] = collect(Hy[ix2,iy1:iy2-1,iz1:iz2-1])
        # x1z:
        fp["incEz_x1z"][:,:,it] = collect(Ez[ix1:ix2-1,iy1,iz1:iz2-1])
        fp["incEx_x1z"][:,:,it] = collect(Ex[ix1:ix2-1,iy1,iz1:iz2-1])
        fp["incHz_x1z"][:,:,it] = collect(Hz[ix1:ix2-1,iy1,iz1:iz2-1])
        fp["incHx_x1z"][:,:,it] = collect(Hx[ix1:ix2-1,iy1,iz1:iz2-1])
        # x2z:
        fp["incEz_x2z"][:,:,it] = collect(Ez[ix1:ix2-1,iy2,iz1:iz2-1])
        fp["incEx_x2z"][:,:,it] = collect(Ex[ix1:ix2-1,iy2,iz1:iz2-1])
        fp["incHz_x2z"][:,:,it] = collect(Hz[ix1:ix2-1,iy2,iz1:iz2-1])
        fp["incHx_x2z"][:,:,it] = collect(Hx[ix1:ix2-1,iy2,iz1:iz2-1])
        # xy1:
        fp["incEy_xy1"][:,:,it] = collect(Ey[ix1:ix2-1,iy1:iy2-1,iz1])
        fp["incEx_xy1"][:,:,it] = collect(Ex[ix1:ix2-1,iy1:iy2-1,iz1])
        fp["incHy_xy1"][:,:,it] = collect(Hy[ix1:ix2-1,iy1:iy2-1,iz1])
        fp["incHx_xy1"][:,:,it] = collect(Hx[ix1:ix2-1,iy1:iy2-1,iz1])
        # xy2:
        fp["incEy_xy2"][:,:,it] = collect(Ey[ix1:ix2-1,iy1:iy2-1,iz2])
        fp["incEx_xy2"][:,:,it] = collect(Ex[ix1:ix2-1,iy1:iy2-1,iz2])
        fp["incHy_xy2"][:,:,it] = collect(Hy[ix1:ix2-1,iy1:iy2-1,iz2])
        fp["incHx_xy2"][:,:,it] = collect(Hx[ix1:ix2-1,iy1:iy2-1,iz2])
    end
    return nothing
end
