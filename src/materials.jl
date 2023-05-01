abstract type Susceptibility end


struct DebyeSusceptibility{T} <: Susceptibility
    depsq :: T
    tauq :: T
end


DebyeSusceptibility(; depsq, tauq) =
    DebyeSusceptibility(promote(depsq, tauq)...)


struct DrudeSusceptibility{T} <: Susceptibility
    wpq :: T
    gammaq :: T
end


DrudeSusceptibility(; wpq, gammaq) =
    DrudeSusceptibility(promote(wpq, gammaq)...)


struct LorentzSusceptibility{T} <: Susceptibility
    depsq :: T
    wq :: T
    deltaq :: T
end


LorentzSusceptibility(; depsq, wq, deltaq) =
    LorentzSusceptibility(promote(depsq, wq, deltaq)...)


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
    (; depsq, tauq) = chi
    aq = 1 / tauq
    bq = EPS0 * depsq / tauq
    Aq = 1 - aq * dt
    Bq = 0.0
    Cq = bq * dt
    return Aq, Bq, Cq
end


function ade_coefficients(chi::DrudeSusceptibility, dt)
    (; wpq, gammaq) = chi
    aq = gammaq
    bq = EPS0 * wpq^2
    Aq = 2 / (aq * dt / 2 + 1)
    Bq = (aq * dt / 2 - 1) / (aq * dt / 2 + 1)
    Cq = bq * dt^2 / (aq * dt / 2 + 1)
    return Aq, Bq, Cq
end


function ade_coefficients(chi::LorentzSusceptibility, dt)
    (; depsq, wq, deltaq) = chi
    aq = 2 * deltaq
    bq = wq^2
    cq = EPS0 * depsq * wq^2
    Aq = (2 - bq * dt^2) / (aq * dt / 2 + 1)
    Bq = (aq * dt / 2 - 1) / (aq * dt / 2 + 1)
    Cq = cq * dt^2 / (aq * dt / 2 + 1)
    return Aq, Bq, Cq
end
