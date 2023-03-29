struct Source{A, F, P, S}
    Amp :: A
    tdel :: A
    waveform :: F
    p :: P
    Fcomp :: S
end

@adapt_structure Source


function add_source!(field, source, t)
    (; Amp, tdel, waveform, p, Fcomp) = source
    F = getfield(field, Fcomp)
    @. F += Amp * waveform(t - tdel, (p,))
    return nothing
end
