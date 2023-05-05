struct Source{A, F, P, S}
    Amp :: A
    Phi :: A
    waveform :: F
    p :: P
    component :: S
end

@adapt_structure Source


function add_source!(field, source, t)
    (; Amp, Phi, waveform, p, component) = source
    F = getfield(field, component)
    @. F += Amp * waveform(t - Phi, (p,))
    return nothing
end


# ******************************************************************************
function PointSource(
    grid::Grid1D;
    position, component, frequency, amplitude=1, phase=0, waveform, p=(),
)
    (; Nz, z) = grid

    zsrc = position
    izsrc = searchsortedfirst(z, zsrc)

    Amp, Phi = zeros(Nz), zeros(Nz)
    Amp[izsrc] = amplitude
    Phi[izsrc] = phase / frequency

    return Source(Amp, Phi, waveform, p, component)
end


function PointSource(
    grid::Grid2D;
    position, component, frequency, amplitude=1, phase=0, waveform, p=(),
)
    (; Nx, Nz, x, z) = grid

    xsrc, zsrc = position
    ixsrc = searchsortedfirst(x, xsrc)
    izsrc = searchsortedfirst(z, zsrc)

    Amp, Phi = zeros(Nx,Nz), zeros(Nx,Nz)
    Amp[ixsrc,izsrc] = amplitude
    Phi[ixsrc,izsrc] = phase / frequency

    return Source(Amp, Phi, waveform, p, component)
end


function PointSource(
    grid::Grid3D;
    position, component, frequency, amplitude=1, phase=0, waveform, p=(),
)
    (; Nx, Ny, Nz, x, y, z) = grid

    xsrc, ysrc, zsrc = position
    ixsrc = searchsortedfirst(x, xsrc)
    iysrc = searchsortedfirst(y, ysrc)
    izsrc = searchsortedfirst(z, zsrc)

    Amp, Phi = zeros(Nx,Ny,Nz), zeros(Nx,Ny,Nz)
    Amp[ixsrc,iysrc,izsrc] = amplitude
    Phi[ixsrc,iysrc,izsrc] = phase / frequency

    return Source(Amp, Phi, waveform, p, component)
end


# ******************************************************************************
function LineSource(
    grid::Grid2D;
    position, component, frequency, amplitude, phase, waveform, p=(),
)
    (; Nx, Nz, x, z) = grid

    zsrc = position
    izsrc = searchsortedfirst(z, zsrc)

    Amp, Phi = zeros(Nx,Nz), zeros(Nx,Nz)
    for ix=1:Nx
        Amp[ix,izsrc] = amplitude(x[ix])
        Phi[ix,izsrc] = phase(x[ix]) / frequency
    end

    return Source(Amp, Phi, waveform, p, component)
end


# ******************************************************************************
function PlaneSource(
    grid::Grid3D;
    position, component, frequency, amplitude, phase, waveform, p=(),
)
    (; Nx, Ny, Nz, x, y, z) = grid

    zsrc = position
    izsrc = searchsortedfirst(z, zsrc)

    Amp, Phi = zeros(Nx,Ny,Nz), zeros(Nx,Ny,Nz)
    for iy=1:Ny, ix=1:Nx
        Amp[ix,iy,izsrc] = amplitude(x[ix], y[iy])
        Phi[ix,iy,izsrc] = phase(x[ix], y[iy]) / frequency
    end

    return Source(Amp, Phi, waveform, p, component)
end
