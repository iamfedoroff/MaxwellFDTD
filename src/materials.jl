struct Material{G, T, C}
    geometry :: G
    eps :: T
    mu :: T
    sigma :: T
    chi :: C
end


function Material(; geometry, eps=1, mu=1, sigma=0, chi=nothing)
    eps, mu, sigma = promote(eps, mu, sigma)
    return Material(geometry, eps, mu, sigma, chi)
end


# ******************************************************************************************
# Susceptibilities
# ******************************************************************************************
abstract type Susceptibility end


struct DebyeSusceptibility{T} <: Susceptibility
    deps :: T
    tau :: T
end


function DebyeSusceptibility(; deps, tau)
    return DebyeSusceptibility(promote(deps, tau)...)
end


function ade_coefficients(chi::DebyeSusceptibility, dt)
    (; deps, tau) = chi
    aq = 1 / tau
    bq = EPS0 * deps / tau
    Aq = 1 - aq * dt
    Bq = 0.0
    Cq = bq * dt
    return Aq, Bq, Cq
end


# ------------------------------------------------------------------------------------------
struct DrudeSusceptibility{T} <: Susceptibility
    wp :: T
    gamma :: T
end


function DrudeSusceptibility(; wp, gamma)
    return DrudeSusceptibility(promote(wp, gamma)...)
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


# ------------------------------------------------------------------------------------------
struct LorentzSusceptibility{T} <: Susceptibility
    deps :: T
    w0 :: T
    delta :: T
end


function LorentzSusceptibility(; deps, w0, delta)
    return LorentzSusceptibility(promote(deps, w0, delta)...)
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
