abstract type Susceptibility end


struct DebyeSusceptibility{T} <: Susceptibility
    depsq :: T
    tauq :: T
end
DebyeSusceptibility(; depsq, tauq) = DebyeSusceptibility(depsq, tauq)


struct DrudeSusceptibility{T} <: Susceptibility
    wpq :: T
    gammaq :: T
end
DrudeSusceptibility(; wq, gammaq) = DrudeSusceptibility(wq, gammaq)


struct LorentzSusceptibility{T} <: Susceptibility
    depsq :: T
    wq :: T
    deltaq :: T
end
LorentzSusceptibility(; depsq, wq, deltaq) =
    LorentzSusceptibility(depsq, wq, deltaq)


struct LorentzMultiSusceptibility{A} <: Susceptibility
    depsq :: A
    wq :: A
    deltaq :: A
end
    LorentzMultiSusceptibility(; depsq, wq, deltaq) =
        LorentzMultiSusceptibility(depsq, wq, deltaq)


struct DrudeLorentzSusceptibility{T, A} <: Susceptibility
    wpq :: T
    gammaq :: T
    depsq :: A
    wq :: A
    deltaq :: A
end
DrudeLorentzSusceptibility(; wpq, gammaq, depsq, wq, deltaq) =
    DrudeLorentzSusceptibility(wpq, gammaq, depsq, wq, deltaq)


# ******************************************************************************
struct Material{T, C}
    eps :: T
    mu :: T
    sigma :: T
    chi :: C
end
Material(; eps, mu, sigma, chi) = Material(eps, mu, sigma, chi)
