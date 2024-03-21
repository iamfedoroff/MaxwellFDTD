struct DFT{A, T}
    Nw :: Int
    w :: A
    tshift :: T
    dt :: T
    issum :: Bool
end


function DFT(t; wmin=nothing, wmax=nothing, sum=false)
    Nt = length(t)
    dt = t[2] - t[1]

    tshift = iseven(Nt) ? div(Nt,2)*dt : div(Nt+1,2)*dt

    Nw = div(Nt,2) + 1
    w = [2*pi * (iw-1) / (dt * Nt) for iw=1:Nw]
    iwmin = isnothing(wmin) ? 1 : argmin(abs.(w .- wmin))
    iwmax = isnothing(wmax) ? Nw : argmin(abs.(w .- wmax))
    w = w[iwmin:iwmax]
    Nw = length(w)

    return DFT(Nw, w, tshift, dt, sum)
end


function (dft::DFT)(S, F, t)
    (; w, tshift, dt) = dft
    @inbounds for iw in eachindex(w)
        S[iw] += 2 * F * exp(-1im * w[iw] * (t - tshift)) * dt
    end
    return nothing
end


function (dft::DFT)(S, F::AbstractArray, t)
    (; w, tshift, dt, issum) = dft
    if issum
        @inbounds for iw in eachindex(w)
            S[iw] += 2 * sum(F * exp(-1im * w[iw] * (t - tshift))) * dt
        end
    else
        @inbounds for iw in eachindex(w)
            tmp = 2 * exp(-1im * w[iw] * (t - tshift)) * dt
            for j in eachindex(F)
                S[j,iw] += F[j] * tmp
            end
        end
    end
    return nothing
end
