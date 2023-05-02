abstract type Susceptibility end


struct DebyeSusceptibility{T} <: Susceptibility
    deps :: T
    tau :: T
end


DebyeSusceptibility(; deps, tau) =
    DebyeSusceptibility(promote(deps, tau)...)


struct DrudeSusceptibility{T} <: Susceptibility
    wp :: T
    gamma :: T
end


DrudeSusceptibility(; wp, gamma) =
    DrudeSusceptibility(promote(wp, gamma)...)


struct LorentzSusceptibility{T} <: Susceptibility
    deps :: T
    w0 :: T
    delta :: T
end


LorentzSusceptibility(; deps, w0, delta) =
    LorentzSusceptibility(promote(deps, w0, delta)...)


# ******************************************************************************
struct Material{T, C}
    eps :: T
    mu :: T
    sigma :: T
    chi :: C
end


function Material(
    ; eps, mu, sigma, chi::Union{Susceptibility,Vector{<:Susceptibility}},
)
    if chi isa Susceptibility
        chi = [chi]
    end
    return Material(promote(eps, mu, sigma)..., chi)
end


# ******************************************************************************
function ade_coefficients(chi::DebyeSusceptibility, dt)
    (; deps, tau) = chi
    aq = 1 / tau
    bq = EPS0 * deps / tau
    Aq = 1 - aq * dt
    Bq = 0.0
    Cq = bq * dt
    return Aq, Bq, Cq
end


function ade_coefficients(chi::DrudeSusceptibility, dt)
    (; wp, gamma) = chi
    aq = gamma
    bq = EPS0 * wp^2
    Aq = 2 / (aq * dt / 2 + 1)
    Bq = (aq * dt / 2 - 1) / (aq * dt / 2 + 1)
    Cq = bq * dt^2 / (aq * dt / 2 + 1)
    return Aq, Bq, Cq
end


function ade_coefficients(chi::LorentzSusceptibility, dt)
    (; deps, w0, delta) = chi
    aq = 2 * delta
    bq = w0^2
    cq = EPS0 * deps * w0^2
    Aq = (2 - bq * dt^2) / (aq * dt / 2 + 1)
    Bq = (aq * dt / 2 - 1) / (aq * dt / 2 + 1)
    Cq = cq * dt^2 / (aq * dt / 2 + 1)
    return Aq, Bq, Cq
end
