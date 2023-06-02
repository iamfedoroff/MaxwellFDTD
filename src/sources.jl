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


function add_source!(field, source::SoftSource, t)
    (; isrc, component, Amp, Phi, waveform, p) = source
    F = getfield(field, component)
    @views @. F[isrc] = F[isrc] + Amp * waveform(t - Phi, (p,))
    return nothing
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


function add_source!(field, source::HardSource, t)
    (; isrc, component, Amp, Phi, waveform, p) = source
    F = getfield(field, component)
    @views @. F[isrc] = Amp * waveform(t - Phi, (p,))
    return nothing
end


# ******************************************************************************************
# Util
# ******************************************************************************************
"""
Converts the type of the 'component' field from 'Symbol' to 'Int'
"""
function isbitify(source::Source, field)
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
