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
    incHy_1 :: A   # 1: z=z1
    incEx_1 :: A
    incHy_2 :: A   # 2: z=z2
    incEx_2 :: A
end

@adapt_structure TFSFSource1D


function source_init(data::DataTFSFSource, field::Field1D, t)
    (; fname) = data
    (; grid) = field
    (; z) = grid

    fp = HDF5.h5open(fname, "r")
    tfsf_box = HDF5.read(fp, "tfsf_box")
    tex = HDF5.read(fp, "t")
    incHy_1_ex = HDF5.read(fp, "incHy_1")   # 1: z=z1
    incEx_1_ex = HDF5.read(fp, "incEx_1")
    incHy_2_ex = HDF5.read(fp, "incHy_2")   # 2: z=z2
    incEx_2_ex = HDF5.read(fp, "incEx_2")
    HDF5.close(fp)

    z1, z2 = tfsf_box
    iz1 = argmin(abs.(z .- z1))
    iz2 = argmin(abs.(z .- z2))

    itp(x,y) = linear_interpolation(x, y; extrapolation_bc=Flat())

    # 1: z=z1
    itp_incHy_1 = itp(tex, incHy_1_ex)
    itp_incEx_1 = itp(tex, incEx_1_ex)
    incHy_1 = @. itp_incHy_1(t)
    incEx_1 = @. itp_incEx_1(t)

    # 2: z=z2
    itp_incHy_2 = itp(tex, incHy_2_ex)
    itp_incEx_2 = itp(tex, incEx_2_ex)
    incHy_2 = @. itp_incHy_2(t)
    incEx_2 = @. itp_incEx_2(t)

    return TFSFSource1D(iz1, iz2, incHy_1, incEx_1, incHy_2, incEx_2)
end


function add_source!(model, source::TFSFSource1D, it)
    (; field, dt) = model
    (; iz1, iz2, incHy_1, incEx_1, incHy_2, incEx_2) = source
    (; grid, Hy, Ex, Dx) = field
    (; dz) = grid

    # 1: z=z1
    Hy[iz1-1] += dt / (MU0*dz) * incEx_1[it]
    Ex[iz1] += dt / (EPS0*dz) * incHy_1[it]
    Dx[iz1] += dt / dz * incHy_1[it]

    # 2: z=z2
    Hy[iz2] -= dt / (MU0*dz) * incEx_2[it]
    Ex[iz2] -= dt / (EPS0*dz) * incHy_2[it]
    Dx[iz2] -= dt / dz * incHy_2[it]

    # (; Mh, Me) = model
    # Hy[iz1-1] += Mh[iz1]/dz * incEx_1[it]
    # Ex[iz1] += Me[iz1]*dt/dz * incHy_1[it]
    # Dx[iz1] += EPS0 * Me[iz1]*dt/dz * incHy_1[it]

    # Hy[iz2] -= Mh[iz2]/dz * incEx_2[it]
    # Ex[iz2] -= Me[iz2]*dt/dz * incHy_2[it]
    # Dx[iz2] -= EPS0 * Me[iz1]*dt/dz * incHy_2[it]

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
    incHy_1z :: A   # 1z: x=x1, z in [z1,z2]
    incEz_1z :: A
    incHy_2z :: A   # 2z: x=x2, z in [z1,z2]
    incEz_2z :: A
    incHy_x1 :: A   # x1: x in [x1,x2], z=z1
    incEx_x1 :: A
    incHy_x2 :: A   # x2: x in [x1,x2], z=z2
    incEx_x2 :: A
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
    incHy_1z_ex = HDF5.read(fp, "incHy_1z")   # 1z: x=x1, z in [z1,z2]
    incEz_1z_ex = HDF5.read(fp, "incEz_1z")
    incHy_2z_ex = HDF5.read(fp, "incHy_2z")   # 2z: x=x2, z in [z1,z2]
    incEz_2z_ex = HDF5.read(fp, "incEz_2z")
    incHy_x1_ex = HDF5.read(fp, "incHy_x1")   # x1: x in [x1,x2], z=z1
    incEx_x1_ex = HDF5.read(fp, "incEx_x1")
    incHy_x2_ex = HDF5.read(fp, "incHy_x2")   # x2: x in [x1,x2], z=z2
    incEx_x2_ex = HDF5.read(fp, "incEx_x2")
    HDF5.close(fp)

    x1, x2, z1, z2 = tfsf_box
    ix1 = argmin(abs.(x .- x1))
    ix2 = argmin(abs.(x .- x2))
    iz1 = argmin(abs.(z .- z1))
    iz2 = argmin(abs.(z .- z2))

    Nt = length(t)

    itp(x,y) = linear_interpolation(x, y; extrapolation_bc=Flat())

    # 1z: x=x1, z in [z1,z2]
    itp_incHy_1z = itp((zex,tex), incHy_1z_ex)
    itp_incEz_1z = itp((zex,tex), incEz_1z_ex)
    incHy_1z = [itp_incHy_1z(z[iz], t[it]) for iz=iz1:iz2-1, it=1:Nt]
    incEz_1z = [itp_incEz_1z(z[iz], t[it]) for iz=iz1:iz2-1, it=1:Nt]

    # 2z: x=x2, z in [z1,z2]
    itp_incHy_2z = itp((zex,tex), incHy_2z_ex)
    itp_incEz_2z = itp((zex,tex), incEz_2z_ex)
    incHy_2z = [itp_incHy_2z(z[iz], t[it]) for iz=iz1:iz2-1, it=1:Nt]
    incEz_2z = [itp_incEz_2z(z[iz], t[it]) for iz=iz1:iz2-1, it=1:Nt]

    # x1: x in [x1,x2], z=z1
    itp_incHy_x1 = itp((xex,tex), incHy_x1_ex)
    itp_incEx_x1 = itp((xex,tex), incEx_x1_ex)
    incHy_x1 = [itp_incHy_x1(x[ix], t[it]) for ix=ix1:ix2-1, it=1:Nt]
    incEx_x1 = [itp_incEx_x1(x[ix], t[it]) for ix=ix1:ix2-1, it=1:Nt]

    # x2: x in [x1,x2], z=z2
    itp_incHy_x2 = itp((xex,tex), incHy_x2_ex)
    itp_incEx_x2 = itp((xex,tex), incEx_x2_ex)
    incHy_x2 = [itp_incHy_x2(x[ix], t[it]) for ix=ix1:ix2-1, it=1:Nt]
    incEx_x2 = [itp_incEx_x2(x[ix], t[it]) for ix=ix1:ix2-1, it=1:Nt]

    return TFSFSource2D(
        ix1, ix2, iz1, iz2,
        incHy_1z, incEz_1z, incHy_2z, incEz_2z, incHy_x1, incEx_x1, incHy_x2, incEx_x2,
    )
end


function add_source!(model, source::TFSFSource2D, it)
    (; field, dt) = model
    (; ix1, ix2, iz1, iz2,
       incHy_1z, incEz_1z, incHy_2z, incEz_2z,
       incHy_x1, incEx_x1, incHy_x2, incEx_x2) = source
    (; grid, Hy, Dx, Dz, Ex, Ez) = field
    (; dx, dz) = grid

    # 1z: x=x1, z in [z1,z2]
    @views @. Hy[ix1-1,iz1:iz2-1] -= dt / (MU0*dx) * incEz_1z[:,it]
    @views @. Ez[ix1,iz1:iz2-1] -= dt / (EPS0*dx) * incHy_1z[:,it]
    @views @. Dz[ix1,iz1:iz2-1] -= dt / dx * incHy_1z[:,it]

    # 2z: x=x2, z in [z1,z2]
    @views @. Hy[ix2,iz1:iz2-1] += dt / (MU0*dx) * incEz_2z[:,it]
    @views @. Ez[ix2,iz1:iz2-1] += dt / (EPS0*dx) * incHy_2z[:,it]
    @views @. Dz[ix2,iz1:iz2-1] += dt / dx * incHy_2z[:,it]

    # x1: x in [x1,x2], z=z1
    @views @. Hy[ix1:ix2-1,iz1-1] += dt / (MU0*dz) * incEx_x1[:,it]
    @views @. Ex[ix1:ix2-1,iz1] += dt / (EPS0*dz) * incHy_x1[:,it]
    @views @. Dx[ix1:ix2-1,iz1] += dt / dz * incHy_x1[:,it]

    # x2: x in [x1,x2], z=z2
    @views @. Hy[ix1:ix2-1,iz2] -= dt / (MU0*dz) * incEx_x2[:,it]
    @views @. Ex[ix1:ix2-1,iz2] -= dt / (EPS0*dz) * incHy_x2[:,it]
    @views @. Dx[ix1:ix2-1,iz2] -= dt / dz * incHy_x2[:,it]
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
    incHy_1yz :: A   # 1yz: x=x1, y in [y1,y2], z in [z1,z2]
    incHz_1yz :: A
    incEy_1yz :: A
    incEz_1yz :: A
    incHy_2yz :: A   # 2yz: x=x2, y in [y1,y2], z in [z1, z2]
    incHz_2yz :: A
    incEy_2yz :: A
    incEz_2yz :: A
    incHx_x1z :: A   # x1z: x in [x1,x2], y=y1, z in [z1,z2]
    incHz_x1z :: A
    incEx_x1z :: A
    incEz_x1z :: A
    incHx_x2z :: A   # x2z: x in [x1,x2], y=y2, z in [z1,z2]
    incHz_x2z :: A
    incEx_x2z :: A
    incEz_x2z :: A
    incHx_xy1 :: A   # xy1: x in [x1,x2], y in [y1,y2], z=z1
    incHy_xy1 :: A
    incEx_xy1 :: A
    incEy_xy1 :: A
    incHx_xy2 :: A   # xy2: x in [x1,x2], y in [y1,y2], z=z2
    incHy_xy2 :: A
    incEx_xy2 :: A
    incEy_xy2 :: A
end

@adapt_structure TFSFSource3D


function source_init(data::DataTFSFSource, field::Field3D, t)
    (; fname) = data
    (; grid) = field
    (; x, y, z) = grid

    fp = HDF5.h5open(fname, "r")
    tfsf_box = HDF5.read(fp, "tfsf_box")
    xex = HDF5.read(fp, "x")
    yex = HDF5.read(fp, "y")
    zex = HDF5.read(fp, "z")
    tex = HDF5.read(fp, "t")
    incHy_1yz_ex = HDF5.read(fp, "incHy_1yz")   # 1yz: x=x1, y in [y1,y2], z in [z1,z2]
    incHz_1yz_ex = HDF5.read(fp, "incHz_1yz")
    incEy_1yz_ex = HDF5.read(fp, "incEy_1yz")
    incEz_1yz_ex = HDF5.read(fp, "incEz_1yz")
    incHy_2yz_ex = HDF5.read(fp, "incHy_2yz")   # 2yz: x=x2, y in [y1,y2], z in [z1, z2]
    incHz_2yz_ex = HDF5.read(fp, "incHz_2yz")
    incEy_2yz_ex = HDF5.read(fp, "incEy_2yz")
    incEz_2yz_ex = HDF5.read(fp, "incEz_2yz")
    incHx_x1z_ex = HDF5.read(fp, "incHx_x1z")   # x1z: x in [x1,x2], y=y1, z in [z1,z2]
    incHz_x1z_ex = HDF5.read(fp, "incHz_x1z")
    incEx_x1z_ex = HDF5.read(fp, "incEx_x1z")
    incEz_x1z_ex = HDF5.read(fp, "incEz_x1z")
    incHx_x2z_ex = HDF5.read(fp, "incHx_x2z")   # x2z: x in [x1,x2], y=y2, z in [z1,z2]
    incHz_x2z_ex = HDF5.read(fp, "incHz_x2z")
    incEx_x2z_ex = HDF5.read(fp, "incEx_x2z")
    incEz_x2z_ex = HDF5.read(fp, "incEz_x2z")
    incHx_xy1_ex = HDF5.read(fp, "incHx_xy1")   # xy1: x in [x1,x2], y in [y1,y2], z=z1
    incHy_xy1_ex = HDF5.read(fp, "incHy_xy1")
    incEx_xy1_ex = HDF5.read(fp, "incEx_xy1")
    incEy_xy1_ex = HDF5.read(fp, "incEy_xy1")
    incHx_xy2_ex = HDF5.read(fp, "incHx_xy2")   # xy2: x in [x1,x2], y in [y1,y2], z=z2
    incHy_xy2_ex = HDF5.read(fp, "incHy_xy2")
    incEx_xy2_ex = HDF5.read(fp, "incEx_xy2")
    incEy_xy2_ex = HDF5.read(fp, "incEy_xy2")
    HDF5.close(fp)

    x1, x2, y1, y2, z1, z2 = tfsf_box
    ix1 = argmin(abs.(x .- x1))
    ix2 = argmin(abs.(x .- x2))
    iy1 = argmin(abs.(y .- y1))
    iy2 = argmin(abs.(y .- y2))
    iz1 = argmin(abs.(z .- z1))
    iz2 = argmin(abs.(z .- z2))

    Nt = length(t)

    itp(x,y) = linear_interpolation(x, y; extrapolation_bc=Flat())

    # 1yz: x=x1, y in [y1,y2], z in [z1,z2]
    itp_incHy_1yz = itp((yex, zex[1:end-1], tex), incHy_1yz_ex)
    itp_incHz_1yz = itp((yex[1:end-1], zex, tex), incHz_1yz_ex)
    itp_incEy_1yz = itp((yex[1:end-1], zex, tex), incEy_1yz_ex)
    itp_incEz_1yz = itp((yex, zex[1:end-1], tex), incEz_1yz_ex)
    incHy_1yz = [itp_incHy_1yz(y[iy], z[iz], t[it]) for iy=iy1:iy2, iz=iz1:iz2-1, it=1:Nt]
    incHz_1yz = [itp_incHz_1yz(y[iy], z[iz], t[it]) for iy=iy1:iy2-1, iz=iz1:iz2, it=1:Nt]
    incEy_1yz = [itp_incEy_1yz(y[iy], z[iz], t[it]) for iy=iy1:iy2-1, iz=iz1:iz2, it=1:Nt]
    incEz_1yz = [itp_incEz_1yz(y[iy], z[iz], t[it]) for iy=iy1:iy2, iz=iz1:iz2-1, it=1:Nt]

    # 2yz: x=x2, y in [y1,y2], z in [z1, z2]
    itp_incHy_2yz = itp((yex, zex[1:end-1], tex), incHy_2yz_ex)
    itp_incHz_2yz = itp((yex[1:end-1], zex, tex), incHz_2yz_ex)
    itp_incEy_2yz = itp((yex[1:end-1], zex, tex), incEy_2yz_ex)
    itp_incEz_2yz = itp((yex, zex[1:end-1], tex), incEz_2yz_ex)
    incHy_2yz = [itp_incHy_2yz(y[iy], z[iz], t[it]) for iy=iy1:iy2, iz=iz1:iz2-1, it=1:Nt]
    incHz_2yz = [itp_incHz_2yz(y[iy], z[iz], t[it]) for iy=iy1:iy2-1, iz=iz1:iz2, it=1:Nt]
    incEy_2yz = [itp_incEy_2yz(y[iy], z[iz], t[it]) for iy=iy1:iy2-1, iz=iz1:iz2, it=1:Nt]
    incEz_2yz = [itp_incEz_2yz(y[iy], z[iz], t[it]) for iy=iy1:iy2, iz=iz1:iz2-1, it=1:Nt]

    # x1z: x in [x1,x2], y=y1, z in [z1,z2]
    itp_incHx_x1z = itp((xex, zex[1:end-1], tex), incHx_x1z_ex)
    itp_incHz_x1z = itp((xex[1:end-1], zex, tex), incHz_x1z_ex)
    itp_incEx_x1z = itp((xex[1:end-1], zex, tex), incEx_x1z_ex)
    itp_incEz_x1z = itp((xex, zex[1:end-1], tex), incEz_x1z_ex)
    incHx_x1z = [itp_incHx_x1z(x[ix], z[iz], t[it]) for ix=ix1:ix2, iz=iz1:iz2-1, it=1:Nt]
    incHz_x1z = [itp_incHz_x1z(x[ix], z[iz], t[it]) for ix=ix1:ix2-1, iz=iz1:iz2, it=1:Nt]
    incEx_x1z = [itp_incEx_x1z(x[ix], z[iz], t[it]) for ix=ix1:ix2-1, iz=iz1:iz2, it=1:Nt]
    incEz_x1z = [itp_incEz_x1z(x[ix], z[iz], t[it]) for ix=ix1:ix2, iz=iz1:iz2-1, it=1:Nt]

    # x2z: x in [x1,x2], y=y2, z in [z1,z2]
    itp_incHx_x2z = itp((xex, zex[1:end-1], tex), incHx_x2z_ex)
    itp_incHz_x2z = itp((xex[1:end-1], zex, tex), incHz_x2z_ex)
    itp_incEx_x2z = itp((xex[1:end-1], zex, tex), incEx_x2z_ex)
    itp_incEz_x2z = itp((xex, zex[1:end-1], tex), incEz_x2z_ex)
    incHx_x2z = [itp_incHx_x2z(x[ix], z[iz], t[it]) for ix=ix1:ix2, iz=iz1:iz2-1, it=1:Nt]
    incHz_x2z = [itp_incHz_x2z(x[ix], z[iz], t[it]) for ix=ix1:ix2-1, iz=iz1:iz2, it=1:Nt]
    incEx_x2z = [itp_incEx_x2z(x[ix], z[iz], t[it]) for ix=ix1:ix2-1, iz=iz1:iz2, it=1:Nt]
    incEz_x2z = [itp_incEz_x2z(x[ix], z[iz], t[it]) for ix=ix1:ix2, iz=iz1:iz2-1, it=1:Nt]

    # xy1: x in [x1,x2], y in [y1,y2], z=z1
    itp_incHx_xy1 = itp((xex, yex[1:end-1], tex), incHx_xy1_ex)
    itp_incHy_xy1 = itp((xex[1:end-1], yex, tex), incHy_xy1_ex)
    itp_incEx_xy1 = itp((xex[1:end-1], yex, tex), incEx_xy1_ex)
    itp_incEy_xy1 = itp((xex, yex[1:end-1], tex), incEy_xy1_ex)
    incHx_xy1 = [itp_incHx_xy1(x[ix], y[iy], t[it]) for ix=ix1:ix2, iy=iy1:iy2-1, it=1:Nt]
    incHy_xy1 = [itp_incHy_xy1(x[ix], y[iy], t[it]) for ix=ix1:ix2-1, iy=iy1:iy2, it=1:Nt]
    incEx_xy1 = [itp_incEx_xy1(x[ix], y[iy], t[it]) for ix=ix1:ix2-1, iy=iy1:iy2, it=1:Nt]
    incEy_xy1 = [itp_incEy_xy1(x[ix], y[iy], t[it]) for ix=ix1:ix2, iy=iy1:iy2-1, it=1:Nt]

    # xy2: x in [x1,x2], y in [y1,y2], z=z2
    itp_incHx_xy2 = itp((xex, yex[1:end-1], tex), incHx_xy2_ex)
    itp_incHy_xy2 = itp((xex[1:end-1], yex, tex), incHy_xy2_ex)
    itp_incEx_xy2 = itp((xex[1:end-1], yex, tex), incEx_xy2_ex)
    itp_incEy_xy2 = itp((xex, yex[1:end-1], tex), incEy_xy2_ex)
    incHx_xy2 = [itp_incHx_xy2(x[ix], y[iy], t[it]) for ix=ix1:ix2, iy=iy1:iy2-1, it=1:Nt]
    incHy_xy2 = [itp_incHy_xy2(x[ix], y[iy], t[it]) for ix=ix1:ix2-1, iy=iy1:iy2, it=1:Nt]
    incEx_xy2 = [itp_incEx_xy2(x[ix], y[iy], t[it]) for ix=ix1:ix2-1, iy=iy1:iy2, it=1:Nt]
    incEy_xy2 = [itp_incEy_xy2(x[ix], y[iy], t[it]) for ix=ix1:ix2, iy=iy1:iy2-1, it=1:Nt]

    return TFSFSource3D(
        ix1, ix2, iy1, iy2, iz1, iz2,
        incHy_1yz, incHz_1yz, incEy_1yz, incEz_1yz,
        incHy_2yz, incHz_2yz, incEy_2yz, incEz_2yz,
        incHx_x1z, incHz_x1z, incEx_x1z, incEz_x1z,
        incHx_x2z, incHz_x2z, incEx_x2z, incEz_x2z,
        incHx_xy1, incHy_xy1, incEx_xy1, incEy_xy1,
        incHx_xy2, incHy_xy2, incEx_xy2, incEy_xy2,
    )
end


function add_source!(model, source::TFSFSource3D, it)
    (; field, dt) = model
    (; ix1, ix2, iy1, iy2, iz1, iz2,
       incHy_1yz, incHz_1yz, incEy_1yz, incEz_1yz,
       incHy_2yz, incHz_2yz, incEy_2yz, incEz_2yz,
       incHx_x1z, incHz_x1z, incEx_x1z, incEz_x1z,
       incHx_x2z, incHz_x2z, incEx_x2z, incEz_x2z,
       incHx_xy1, incHy_xy1, incEx_xy1, incEy_xy1,
       incHx_xy2, incHy_xy2, incEx_xy2, incEy_xy2) = source
    (; grid, Hx, Hy, Hz, Dx, Dy, Dz, Ex, Ey, Ez) = field
    (; dx, dy, dz) = grid

    Cm, Ce = dt/MU0, dt/EPS0

    # 1yz: x=x1, y in [y1,y2], z in [z1,z2]
    @views @. Hy[ix1-1,iy1:iy2,iz1:iz2-1] -= Cm/dx * incEz_1yz[:,:,it]
    @views @. Hz[ix1-1,iy1:iy2-1,iz1:iz2] += Cm/dx * incEy_1yz[:,:,it]
    @views @. Ey[ix1,iy1:iy2-1,iz1:iz2] += Ce/dx * incHz_1yz[:,:,it]
    @views @. Ez[ix1,iy1:iy2,iz1:iz2-1] -= Ce/dx * incHy_1yz[:,:,it]
    @views @. Dy[ix1,iy1:iy2-1,iz1:iz2] += EPS0 * Ce/dx * incHz_1yz[:,:,it]
    @views @. Dz[ix1,iy1:iy2,iz1:iz2-1] -= EPS0 * Ce/dx * incHy_1yz[:,:,it]

    # 2yz: x=x2, y in [y1,y2], z in [z1, z2]
    @views @. Hy[ix2,iy1:iy2,iz1:iz2-1] += Cm/dx * incEz_2yz[:,:,it]
    @views @. Hz[ix2,iy1:iy2-1,iz1:iz2] -= Cm/dx * incEy_2yz[:,:,it]
    @views @. Ey[ix2,iy1:iy2-1,iz1:iz2] -= Ce/dx * incHz_2yz[:,:,it]
    @views @. Ez[ix2,iy1:iy2,iz1:iz2-1] += Ce/dx * incHy_2yz[:,:,it]
    @views @. Dy[ix2,iy1:iy2-1,iz1:iz2] -= EPS0 * Ce/dx * incHz_2yz[:,:,it]
    @views @. Dz[ix2,iy1:iy2,iz1:iz2-1] += EPS0 * Ce/dx * incHy_2yz[:,:,it]

    # x1z: x in [x1,x2], y=y1, z in [z1,z2]
    @views @. Hx[ix1:ix2,iy1-1,iz1:iz2-1] += Cm/dy * incEz_x1z[:,:,it]
    @views @. Hz[ix1:ix2-1,iy1-1,iz1:iz2] -= Cm/dy * incEx_x1z[:,:,it]
    @views @. Ex[ix1:ix2-1,iy1,iz1:iz2] -= Ce/dy * incHz_x1z[:,:,it]
    @views @. Ez[ix1:ix2,iy1,iz1:iz2-1] += Ce/dy * incHx_x1z[:,:,it]
    @views @. Dx[ix1:ix2-1,iy1,iz1:iz2] -= EPS0 * Ce/dy * incHz_x1z[:,:,it]
    @views @. Dz[ix1:ix2,iy1,iz1:iz2-1] += EPS0 * Ce/dy * incHx_x1z[:,:,it]

    # x2z: x in [x1,x2], y=y2, z in [z1,z2]
    @views @. Hx[ix1:ix2,iy2,iz1:iz2-1] -= Cm/dy * incEz_x2z[:,:,it]
    @views @. Hz[ix1:ix2-1,iy2,iz1:iz2] += Cm/dy * incEx_x2z[:,:,it]
    @views @. Ex[ix1:ix2-1,iy2,iz1:iz2] += Ce/dy * incHz_x2z[:,:,it]
    @views @. Ez[ix1:ix2,iy2,iz1:iz2-1] -= Ce/dy * incHx_x2z[:,:,it]
    @views @. Dx[ix1:ix2-1,iy2,iz1:iz2] += EPS0 * Ce/dy * incHz_x2z[:,:,it]
    @views @. Dz[ix1:ix2,iy2,iz1:iz2-1] -= EPS0 * Ce/dy * incHx_x2z[:,:,it]

    # xy1: x in [x1,x2], y in [y1,y2], z=z1
    @views @. Hx[ix1:ix2,iy1:iy2-1,iz1-1] -= Cm/dz * incEy_xy1[:,:,it]
    @views @. Hy[ix1:ix2-1,iy1:iy2,iz1-1] += Cm/dz * incEx_xy1[:,:,it]
    @views @. Ex[ix1:ix2-1,iy1:iy2,iz1] += Ce/dz * incHy_xy1[:,:,it]
    @views @. Ey[ix1:ix2,iy1:iy2-1,iz1] -= Ce/dz * incHx_xy1[:,:,it]
    @views @. Dx[ix1:ix2-1,iy1:iy2,iz1] += EPS0 * Ce/dz * incHy_xy1[:,:,it]
    @views @. Dy[ix1:ix2,iy1:iy2-1,iz1] -= EPS0 * Ce/dz * incHx_xy1[:,:,it]

    # xy2: x in [x1,x2], y in [y1,y2], z=z2
    @views @. Hx[ix1:ix2,iy1:iy2-1,iz2] += Cm/dz * incEy_xy2[:,:,it]
    @views @. Hy[ix1:ix2-1,iy1:iy2,iz2] -= Cm/dz * incEx_xy2[:,:,it]
    @views @. Ex[ix1:ix2-1,iy1:iy2,iz2] -= Ce/dz * incHy_xy2[:,:,it]
    @views @. Ey[ix1:ix2,iy1:iy2-1,iz2] += Ce/dz * incHx_xy2[:,:,it]
    @views @. Dx[ix1:ix2-1,iy1:iy2,iz2] -= EPS0 * Ce/dz * incHy_xy2[:,:,it]
    @views @. Dy[ix1:ix2,iy1:iy2-1,iz2] += EPS0 * Ce/dz * incHx_xy2[:,:,it]

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
        HDF5.create_dataset(fp, "incHy_1", T, Nt)   # 1: z=z1
        HDF5.create_dataset(fp, "incEx_1", T, Nt)
        HDF5.create_dataset(fp, "incHy_2", T, Nt)   # 2: z=z2
        HDF5.create_dataset(fp, "incEx_2", T, Nt)
    end

    return DataTFSFSource(tfsf_fname, (iz1, iz2))
end


function write_tfsf_record(model::Model1D, tfsf_data, it)
    (; field) = model
    (; Hy, Ex) = field
    (; fname, tfsf_box) = tfsf_data
    iz1, iz2 = tfsf_box
    HDF5.h5open(fname, "r+") do fp
        fp["incHy_1"][it] = collect(Hy[iz1-1])   # 1: z=z1
        fp["incEx_1"][it] = collect(Ex[iz1])
        fp["incHy_2"][it] = collect(Hy[iz2])   # 2: z=z2
        fp["incEx_2"][it] = collect(Ex[iz2])
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
        HDF5.create_dataset(fp, "incHy_1z", T, (Nzi, Nt))   # 1z: x=x1, z in [z1,z2]
        HDF5.create_dataset(fp, "incEz_1z", T, (Nzi, Nt))
        HDF5.create_dataset(fp, "incHy_2z", T, (Nzi, Nt))   # 2z: x=x2, z in [z1,z2]
        HDF5.create_dataset(fp, "incEz_2z", T, (Nzi, Nt))
        HDF5.create_dataset(fp, "incHy_x1", T, (Nxi, Nt))   # x1: x in [x1,x2], z=z1
        HDF5.create_dataset(fp, "incEx_x1", T, (Nxi, Nt))
        HDF5.create_dataset(fp, "incHy_x2", T, (Nxi, Nt))   # x2: x in [x1,x2], z=z2
        HDF5.create_dataset(fp, "incEx_x2", T, (Nxi, Nt))
    end

    return DataTFSFSource(tfsf_fname, (ix1, ix2, iz1, iz2))
end


function write_tfsf_record(model::Model2D, tfsf_data, it)
    (; field) = model
    (; Hy, Ex, Ez) = field
    (; fname, tfsf_box) = tfsf_data
    ix1, ix2, iz1, iz2 = tfsf_box
    HDF5.h5open(fname, "r+") do fp
        fp["incHy_1z"][:,it] = collect(Hy[ix1-1,iz1:iz2-1])   # 1z: x=x1, z in [z1,z2]
        fp["incEz_1z"][:,it] = collect(Ez[ix1,iz1:iz2-1])
        fp["incHy_2z"][:,it] = collect(Hy[ix2,iz1:iz2-1])   # 2z: x=x2, z in [z1,z2]
        fp["incEz_2z"][:,it] = collect(Ez[ix2,iz1:iz2-1])
        fp["incHy_x1"][:,it] = collect(Hy[ix1:ix2-1,iz1-1])   # x1: x in [x1,x2], z=z1
        fp["incEx_x1"][:,it] = collect(Ex[ix1:ix2-1,iz1])
        fp["incHy_x2"][:,it] = collect(Hy[ix1:ix2-1,iz2])   # x2: x in [x1,x2], z=z2
        fp["incEx_x2"][:,it] = collect(Ex[ix1:ix2-1,iz2])
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
        fp["x"] = collect(x[ix1:ix2])
        fp["y"] = collect(y[iy1:iy2])
        fp["z"] = collect(z[iz1:iz2])
        fp["t"] = collect(t)
        fp["tfsf_box"] = collect(tfsf_box)
        # 1yz: x=x1, y in [y1,y2], z in [z1,z2]
        HDF5.create_dataset(fp, "incHy_1yz", T, (Nyi+1, Nzi, Nt))
        HDF5.create_dataset(fp, "incHz_1yz", T, (Nyi, Nzi+1, Nt))
        HDF5.create_dataset(fp, "incEy_1yz", T, (Nyi, Nzi+1, Nt))
        HDF5.create_dataset(fp, "incEz_1yz", T, (Nyi+1, Nzi, Nt))
        # 2yz: x=x2, y in [y1,y2], z in [z1, z2]
        HDF5.create_dataset(fp, "incHy_2yz", T, (Nyi+1, Nzi, Nt))
        HDF5.create_dataset(fp, "incHz_2yz", T, (Nyi, Nzi+1, Nt))
        HDF5.create_dataset(fp, "incEy_2yz", T, (Nyi, Nzi+1, Nt))
        HDF5.create_dataset(fp, "incEz_2yz", T, (Nyi+1, Nzi, Nt))
        # x1z: x in [x1,x2], y=y1, z in [z1,z2]
        HDF5.create_dataset(fp, "incHx_x1z", T, (Nxi+1, Nzi, Nt))
        HDF5.create_dataset(fp, "incHz_x1z", T, (Nxi, Nzi+1, Nt))
        HDF5.create_dataset(fp, "incEx_x1z", T, (Nxi, Nzi+1, Nt))
        HDF5.create_dataset(fp, "incEz_x1z", T, (Nxi+1, Nzi, Nt))
        # x2z: x in [x1,x2], y=y2, z in [z1,z2]
        HDF5.create_dataset(fp, "incHx_x2z", T, (Nxi+1, Nzi, Nt))
        HDF5.create_dataset(fp, "incHz_x2z", T, (Nxi, Nzi+1, Nt))
        HDF5.create_dataset(fp, "incEx_x2z", T, (Nxi, Nzi+1, Nt))
        HDF5.create_dataset(fp, "incEz_x2z", T, (Nxi+1, Nzi, Nt))
        # xy1: x in [x1,x2], y in [y1,y2], z=z1
        HDF5.create_dataset(fp, "incHx_xy1", T, (Nxi+1, Nyi, Nt))
        HDF5.create_dataset(fp, "incHy_xy1", T, (Nxi, Nyi+1, Nt))
        HDF5.create_dataset(fp, "incEx_xy1", T, (Nxi, Nyi+1, Nt))
        HDF5.create_dataset(fp, "incEy_xy1", T, (Nxi+1, Nyi, Nt))
        # xy2: x in [x1,x2], y in [y1,y2], z=z2
        HDF5.create_dataset(fp, "incHx_xy2", T, (Nxi+1, Nyi, Nt))
        HDF5.create_dataset(fp, "incHy_xy2", T, (Nxi, Nyi+1, Nt))
        HDF5.create_dataset(fp, "incEx_xy2", T, (Nxi, Nyi+1, Nt))
        HDF5.create_dataset(fp, "incEy_xy2", T, (Nxi+1, Nyi, Nt))
    end

    return DataTFSFSource(tfsf_fname, (ix1, ix2, iy1, iy2, iz1, iz2))
end


function write_tfsf_record(model::Model3D, tfsf_data, it)
    (; field) = model
    (; Hx, Hy, Hz, Ex, Ey, Ez) = field
    (; fname, tfsf_box) = tfsf_data
    ix1, ix2, iy1, iy2, iz1, iz2 = tfsf_box

    HDF5.h5open(fname, "r+") do fp
        # 1yz: x=x1, y in [y1,y2], z in [z1,z2]
        fp["incHy_1yz"][:,:,it] = collect(Hy[ix1-1,iy1:iy2,iz1:iz2-1])
        fp["incHz_1yz"][:,:,it] = collect(Hz[ix1-1,iy1:iy2-1,iz1:iz2])
        fp["incEy_1yz"][:,:,it] = collect(Ey[ix1,iy1:iy2-1,iz1:iz2])
        fp["incEz_1yz"][:,:,it] = collect(Ez[ix1,iy1:iy2,iz1:iz2-1])
        # 2yz: x=x2, y in [y1,y2], z in [z1, z2]
        fp["incHy_2yz"][:,:,it] = collect(Hy[ix2,iy1:iy2,iz1:iz2-1])
        fp["incHz_2yz"][:,:,it] = collect(Hz[ix2,iy1:iy2-1,iz1:iz2])
        fp["incEy_2yz"][:,:,it] = collect(Ey[ix2,iy1:iy2-1,iz1:iz2])
        fp["incEz_2yz"][:,:,it] = collect(Ez[ix2,iy1:iy2,iz1:iz2-1])
        # x1z: x in [x1,x2], y=y1, z in [z1,z2]
        fp["incHx_x1z"][:,:,it] = collect(Hx[ix1:ix2,iy1-1,iz1:iz2-1])
        fp["incHz_x1z"][:,:,it] = collect(Hz[ix1:ix2-1,iy1-1,iz1:iz2])
        fp["incEx_x1z"][:,:,it] = collect(Ex[ix1:ix2-1,iy1,iz1:iz2])
        fp["incEz_x1z"][:,:,it] = collect(Ez[ix1:ix2,iy1,iz1:iz2-1])
        # x2z: x in [x1,x2], y=y2, z in [z1,z2]
        fp["incHx_x2z"][:,:,it] = collect(Hx[ix1:ix2,iy2,iz1:iz2-1])
        fp["incHz_x2z"][:,:,it] = collect(Hz[ix1:ix2-1,iy2,iz1:iz2])
        fp["incEx_x2z"][:,:,it] = collect(Ex[ix1:ix2-1,iy2,iz1:iz2])
        fp["incEz_x2z"][:,:,it] = collect(Ez[ix1:ix2,iy2,iz1:iz2-1])
        # xy1: x in [x1,x2], y in [y1,y2], z=z1
        fp["incHx_xy1"][:,:,it] = collect(Hx[ix1:ix2,iy1:iy2-1,iz1-1])
        fp["incHy_xy1"][:,:,it] = collect(Hy[ix1:ix2-1,iy1:iy2,iz1-1])
        fp["incEx_xy1"][:,:,it] = collect(Ex[ix1:ix2-1,iy1:iy2,iz1])
        fp["incEy_xy1"][:,:,it] = collect(Ey[ix1:ix2,iy1:iy2-1,iz1])
        # xy2: x in [x1,x2], y in [y1,y2], z=z2
        fp["incHx_xy2"][:,:,it] = collect(Hx[ix1:ix2,iy1:iy2-1,iz2])
        fp["incHy_xy2"][:,:,it] = collect(Hy[ix1:ix2-1,iy1:iy2,iz2])
        fp["incEx_xy2"][:,:,it] = collect(Ex[ix1:ix2-1,iy1:iy2,iz2])
        fp["incEy_xy2"][:,:,it] = collect(Ey[ix1:ix2,iy1:iy2-1,iz2])
    end
    return nothing
end
