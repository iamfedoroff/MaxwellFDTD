abstract type Source end


struct SoftSource{C, S, A, F, P} <: Source
    isrc :: C
    component :: S
    Amp :: A
    Phi :: A
    waveform :: F
    p :: P
end

@adapt_structure SoftSource


struct HardSource{C, S, A, F, P} <: Source
    isrc :: C
    component :: S
    Amp :: A
    Phi :: A
    waveform :: F
    p :: P
end

@adapt_structure HardSource


# Converts the type of the 'component' field from 'Symbol' to 'Int'
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


function add_source!(field, source::SoftSource, t)
    (; isrc, component, Amp, Phi, waveform, p) = source
    F = getfield(field, component)
    @views @. F[isrc] = F[isrc] + Amp * waveform(t - Phi, (p,))
    return nothing
end


function add_source!(field, source::HardSource, t)
    (; isrc, component, Amp, Phi, waveform, p) = source
    F = getfield(field, component)
    @views @. F[isrc] = Amp * waveform(t - Phi, (p,))
    return nothing
end


# ******************************************************************************
function PointSource(
    grid::Grid1D;
    position, component, frequency, amplitude=1, phase=0, waveform, p=(),
    type=:soft,
)
    (; z) = grid

    zsrc = position
    izsrc = searchsortedfirst(z, zsrc)
    isrc = CartesianIndices((izsrc:izsrc,))

    Amp = amplitude
    Phi = phase / frequency
    Amp, Phi = promote(Amp, Phi)

    if type == :soft
        return SoftSource(isrc, component, Amp, Phi, waveform, p)
    elseif type == :hard
        return HardSource(isrc, component, Amp, Phi, waveform, p)
    else
        error("Wrong source type.")
    end
end


function PointSource(
    grid::Grid2D;
    position, component, frequency, amplitude=1, phase=0, waveform, p=(),
    type=:soft,
)
    (; x, z) = grid

    xsrc, zsrc = position
    ixsrc = searchsortedfirst(x, xsrc)
    izsrc = searchsortedfirst(z, zsrc)
    isrc = CartesianIndices((ixsrc:ixsrc, izsrc:izsrc))

    Amp = amplitude
    Phi = phase / frequency
    Amp, Phi = promote(Amp, Phi)

    if type == :soft
        return SoftSource(isrc, component, Amp, Phi, waveform, p)
    elseif type == :hard
        return HardSource(isrc, component, Amp, Phi, waveform, p)
    else
        error("Wrong source type.")
    end
end


function PointSource(
    grid::Grid3D;
    position, component, frequency, amplitude=1, phase=0, waveform, p=(),
    type=:soft,
)
    (; x, y, z) = grid

    xsrc, ysrc, zsrc = position
    ixsrc = searchsortedfirst(x, xsrc)
    iysrc = searchsortedfirst(y, ysrc)
    izsrc = searchsortedfirst(z, zsrc)
    isrc = CartesianIndices((ixsrc:ixsrc, iysrc:iysrc, izsrc:izsrc))

    Amp = amplitude
    Phi = phase / frequency
    Amp, Phi = promote(Amp, Phi)

    if type == :soft
        return SoftSource(isrc, component, Amp, Phi, waveform, p)
    elseif type == :hard
        return HardSource(isrc, component, Amp, Phi, waveform, p)
    else
        error("Wrong source type.")
    end
end


# ******************************************************************************
function LineSource(
    grid::Grid2D;
    position, component, frequency, amplitude, phase, waveform, p=(),
    type=:soft,
)
    (; Nx, x, z) = grid

    zsrc = position
    izsrc = searchsortedfirst(z, zsrc)
    isrc = CartesianIndices((1:Nx, izsrc:izsrc))

    Amp, Phi = zeros(Nx), zeros(Nx)
    for ix=1:Nx
        Amp[ix] = amplitude(x[ix])
        Phi[ix] = phase(x[ix]) / frequency
    end

    if type == :soft
        return SoftSource(isrc, component, Amp, Phi, waveform, p)
    elseif type == :hard
        return HardSource(isrc, component, Amp, Phi, waveform, p)
    else
        error("Wrong source type.")
    end
end


# ******************************************************************************
function PlaneSource(
    grid::Grid3D;
    position, component, frequency, amplitude, phase, waveform, p=(),
    type=:soft,
)
    (; Nx, Ny, x, y, z) = grid

    zsrc = position
    izsrc = searchsortedfirst(z, zsrc)
    isrc = CartesianIndices((1:Nx, 1:Ny, izsrc:izsrc))

    Amp, Phi = zeros(Nx,Ny), zeros(Nx,Ny)
    for iy=1:Ny, ix=1:Nx
        Amp[ix,iy] = amplitude(x[ix], y[iy])
        Phi[ix,iy] = phase(x[ix], y[iy]) / frequency
    end

    if type == :soft
        return SoftSource(isrc, component, Amp, Phi, waveform, p)
    elseif type == :hard
        return HardSource(isrc, component, Amp, Phi, waveform, p)
    else
        error("Wrong source type.")
    end
end
