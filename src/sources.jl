struct Source{C, S, A, F, P, T}
    isrc :: C
    component :: S
    Amp :: A
    Phi :: A
    waveform :: F
    p :: P
    type :: T
end

@adapt_structure Source


function add_source!(field, source, t)
    (; isrc, component, Amp, Phi, waveform, p, type) = source
    F = getfield(field, component)
    if type == :soft
        @views @. F[isrc] = F[isrc] + Amp * waveform(t - Phi, (p,))
    elseif type == :hard
        @views @. F[isrc] = Amp * waveform(t - Phi, (p,))
    else
        error("Wrong source type.")
    end
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

    return Source(isrc, component, Amp, Phi, waveform, p, type)
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

    return Source(isrc, component, Amp, Phi, waveform, p, type)
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

    return Source(isrc, component, Amp, Phi, waveform, p, type)
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

    return Source(isrc, component, Amp, Phi, waveform, p, type)
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

    return Source(isrc, component, Amp, Phi, waveform, p, type)
end
