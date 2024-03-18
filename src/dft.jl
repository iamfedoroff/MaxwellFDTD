struct DFT{A, T}
    Nw :: Int
    w :: A
    tshift :: T
    dt :: T
end


function DFT(t; wmin=nothing, wmax=nothing)
    Nt = length(t)
    dt = t[2] - t[1]

    tshift = iseven(Nt) ? div(Nt,2)*dt : div(Nt+1,2)*dt

    Nw = div(Nt,2) + 1
    w = [2*pi * (iw-1) / (dt * Nt) for iw=1:Nw]
    iwmin = isnothing(wmin) ? 1 : argmin(abs.(w .- wmin))
    iwmax = isnothing(wmax) ? Nw : argmin(abs.(w .- wmax))
    w = w[iwmin:iwmax]
    Nw = length(w)

    return DFT(Nw, w, tshift, dt)
end


function (dft::DFT)(S, F, t)
    (; tshift, w) = dft
    @inbounds for iw in eachindex(w)
        S[iw] += sum(F * exp(-1im * w[iw] * (t - tshift)))
    end
    return nothing
end
