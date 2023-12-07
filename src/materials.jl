struct Material{G, T, C, C2, C3, P}
    geometry :: G
    eps :: T
    mu :: T
    sigma :: T
    chi :: C
    chi2 :: C2
    chi3 :: C3
    plasma :: P
end


"""
    Material(; kwargs...)

Material position and properties.

# Keywords
- `geometry::Union{Function,AbstractArray}`: geometry function or array
- `eps::Real=1`: permittivity
- `mu::Real=1`: permeability
- `sigma::Real=0`: conductivity
- `chi::Susceptibility=nothing`: linear susceptibility
- `chi2::Real=nothing`: second-order nonlinear susceptibility
- `chi3::Real=nothing`: third-order nonlinear susceptibility
- `plasma::Plasma=nothing`: plasma data
"""
function Material(;
    geometry, eps=1, mu=1, sigma=0, chi=nothing, chi2=nothing, chi3=nothing, plasma=nothing,
)
    if eps < 0
        error(
            "You specified negative eps=$eps." *
            " Negative frequency-independent permittivities lead to unstable solutions." *
            " If you want to model a material with negative permittivity at a given" *
            " wavelength, use frequency-dependent susceptibility chi."
        )
    end
    eps, mu, sigma = promote(eps, mu, sigma)
    return Material(geometry, eps, mu, sigma, chi, chi2, chi3, plasma)
end


# ******************************************************************************************
struct Plasma{T, F}
    ionrate :: F
    rho0 :: T
    nuc :: T
    frequency :: T
    Uiev :: T
    mr :: T
end


function Plasma(; ionrate, rho0, nuc, frequency, Uiev, mr=1)
    rho0, nuc, frequency, Uiev, mr = promote(rho0, nuc, frequency, Uiev, mr)
    return Plasma(ionrate, rho0, nuc, frequency, Uiev, mr)
end


function ade_plasma_coefficients(nuc, mr, dt)
    Ap = 2 / (nuc*dt/2 + 1)
    Bp = (nuc*dt/2 - 1) / (nuc*dt/2 + 1)
    Cp = QE^2 / (ME*mr) * dt^2 / (nuc*dt/2 + 1)
    return Ap, Bp, Cp
end


# ******************************************************************************************
# Materials
# ******************************************************************************************
struct MaterialStruct1D{G, T, V, A, B, F}
    geometry :: G
    sigma :: T   # conductivity
    # Update coefficients for H, E and D fields:
    Mh :: T
    Me :: T
    Md1 :: T
    Md2 :: T
    # Linear dispersion:
    isdispersion :: Bool
    Aq :: V
    Bq :: V
    Cq :: V
    Px :: A
    oldPx1 :: A
    oldPx2 :: A
    # Kerr:
    iskerr :: Bool
    Mk2 :: T
    Mk3 :: T
    # Plasma:
    isplasma :: Bool
    ionrate :: F
    Rava :: T
    ksi :: T   # field to intensity conversion coefficient
    rho0 :: T
    rho :: B   # electron density
    drho :: B   # time derivative of electron density
    Ap :: T
    Bp :: T
    Cp :: T
    Ppx :: B
    oldPpx1 :: B
    oldPpx2 :: B
    Ma :: T
    Pax :: B
end

@adapt_structure MaterialStruct1D


function MaterialStruct(material, grid::Grid1D, dt)
    if isnothing(material)
        material = Material(geometry = z -> false)
    end
    (; geometry, eps, mu, sigma, chi, chi2, chi3, plasma) = material
    (; Nz, z) = grid

    if typeof(geometry) <: Function
        geometry = [Bool(geometry(z[iz])) for iz=1:Nz]
    else
        geometry = [Bool(geometry[iz]) for iz=1:Nz]
    end

    sigma = Float64(sigma)
    sigmaD = sigma / (EPS0*eps)   # J=sigma*E -> J=sigmaD*D

    # Update coefficients for H, E and D fields:
    Mh = dt / (MU0*mu)
    Me = 1 / (EPS0*eps)
    Md1 = (1 - sigmaD*dt/2) / (1 + sigmaD*dt/2)
    Md2 = dt / (1 + sigmaD*dt/2)

    # Variables for ADE dispersion calculation:
    if isnothing(chi)
        isdispersion = false

        # to avoid issues with CUDA kernels, the variables should have the same type for all
        # logical branches:
        Aq, Bq, Cq = (zeros(1) for i=1:3)
        Px, oldPx1, oldPx2 = (zeros(1) for i=1:3)
    else
        isdispersion = true

        if chi isa Susceptibility
            chi = (chi,)
        end

        Nq = length(chi)
        Aq, Bq, Cq = (zeros(Nq) for i=1:3)
        for iq=1:Nq
            Aq[iq], Bq[iq], Cq[iq] = ade_coefficients(chi[iq], dt)
        end
        Px, oldPx1, oldPx2 = (zeros(Nq,Nz) for i=1:3)
    end

    # Kerr:
    if isnothing(chi2) && isnothing(chi3)
        iskerr = false
        Mk2, Mk3 = 0.0, 0.0
    else
        iskerr = true
        isnothing(chi2) ? Mk2 = 0.0 : Mk2 = EPS0*chi2
        isnothing(chi3) ? Mk3 = 0.0 : Mk3 = EPS0*chi3
    end

    # Plasma:
    if isnothing(plasma)
        isplasma = false
        ionrate = identity
        Rava, ksi, rho0, Ap, Bp, Cp, Ma = (0.0 for i=1:7)
        rho, drho = (zeros(1) for i=1:2)
        Ppx, oldPpx1, oldPpx2, Pax = (zeros(1) for i=1:4)
    else
        isplasma = true
        (; ionrate, rho0, nuc, frequency, Uiev, mr) = plasma

        # Field to intensity conversion coefficient (I = ksi*|E|^2)
        if isdispersion
            chi0 = susceptibility(chi, frequency)
        else
            chi0 = 0
        end
        n = sqrt((eps + real(chi0)) * mu)   # refractive index
        ksi = n * EPS0 * C0 / 2   # 1/2 from <cos^2(t)>

        Ap, Bp, Cp = ade_plasma_coefficients(nuc, mr, dt)
        Ui = Uiev * QE   # eV -> J
        Wph = HBAR * frequency   # energy of one photon
        K = ceil(Ui / Wph)   # minimum number of photons to extract one electron
        sigmaB = QE^2 / (ME*mr) * nuc / (nuc^2 + frequency^2)
        Rava = sigmaB / Ui
        Ma = K * Wph * dt   # update coefficient
        rho, drho, Ppx, oldPpx1, oldPpx2, Pax = (zeros(Nz) for i=1:6)
    end

    return MaterialStruct1D(
        geometry, sigma, Mh, Me, Md1, Md2, isdispersion, Aq, Bq, Cq, Px, oldPx1, oldPx2,
        iskerr, Mk2, Mk3, isplasma, ionrate, Rava, ksi, rho0, rho, drho, Ap, Bp, Cp, Ppx,
        oldPpx1, oldPpx2, Ma, Pax,
    )
end


# ------------------------------------------------------------------------------------------
struct MaterialStruct2D{G, T, V, A, B, F}
    geometry :: G
    sigma :: T   # conductivity
    # Update coefficients for H, E and D fields:
    Mh :: T
    Me :: T
    Md1 :: T
    Md2 :: T
    # Linear dispersion:
    isdispersion :: Bool
    Aq :: V
    Bq :: V
    Cq :: V
    Px :: A
    oldPx1 :: A
    oldPx2 :: A
    Pz :: A
    oldPz1 :: A
    oldPz2 :: A
    # Kerr:
    iskerr :: Bool
    Mk2 :: T
    Mk3 :: T
    # Plasma:
    isplasma :: Bool
    ionrate :: F
    Rava :: T
    ksi :: T   # field to intensity conversion coefficient
    rho0 :: T
    rho :: B   # electron density
    drho :: B   # time derivative of electron density
    Ap :: T
    Bp :: T
    Cp :: T
    Ppx :: B
    oldPpx1 :: B
    oldPpx2 :: B
    Ppz :: B
    oldPpz1 :: B
    oldPpz2 :: B
    Ma :: T
    Pax :: B
    Paz :: B
end

@adapt_structure MaterialStruct2D


function MaterialStruct(material, grid::Grid2D, dt)
    if isnothing(material)
        material = Material(geometry = (x,z) -> false)
    end
    (; geometry, eps, mu, sigma, chi, chi2, chi3, plasma) = material
    (; Nx, Nz, x, z) = grid

    if typeof(geometry) <: Function
        geometry = [Bool(geometry(x[ix],z[iz])) for ix=1:Nx, iz=1:Nz]
    else
        geometry = [Bool(geometry[ix,iz]) for ix=1:Nx, iz=1:Nz]
    end

    sigma = Float64(sigma)
    sigmaD = sigma / (EPS0*eps)   # J=sigma*E -> J=sigma*D

    # Update coefficients for H, E and D fields:
    Mh = dt / (MU0*mu)
    Me = 1 / (EPS0*eps)
    Md1 = (1 - sigmaD*dt/2) / (1 + sigmaD*dt/2)
    Md2 = dt / (1 + sigmaD*dt/2)

    # Variables for ADE dispersion calculation:
    if isnothing(chi)
        isdispersion = false

        # to avoid issues with CUDA kernels, the types of variables should be the same
        # for all logical branches:
        Aq, Bq, Cq = (zeros(1) for i=1:3)
        Px, oldPx1, oldPx2 = (zeros(1) for i=1:3)
        Pz, oldPz1, oldPz2 = (zeros(1) for i=1:3)
    else
        isdispersion = true

        if chi isa Susceptibility
            chi = (chi,)
        end

        Nq = length(chi)
        Aq, Bq, Cq = (zeros(Nq) for i=1:3)
        for iq=1:Nq
            Aq[iq], Bq[iq], Cq[iq] = ade_coefficients(chi[iq], dt)
        end
        Px, oldPx1, oldPx2 = (zeros(Nq,Nx,Nz) for i=1:3)
        Pz, oldPz1, oldPz2 = (zeros(Nq,Nx,Nz) for i=1:3)
    end

    # Kerr:
    if isnothing(chi2) && isnothing(chi3)
        iskerr = false
        Mk2, Mk3 = 0.0, 0.0
    else
        iskerr = true
        isnothing(chi2) ? Mk2 = 0.0 : Mk2 = EPS0*chi2
        isnothing(chi3) ? Mk3 = 0.0 : Mk3 = EPS0*chi3
    end

    # Plasma:
    if isnothing(plasma)
        isplasma = false
        ionrate = identity
        Rava, ksi, rho0, Ap, Bp, Cp, Ma = (0.0 for i=1:7)
        rho, drho = (zeros(1) for i=1:2)
        Ppx, oldPpx1, oldPpx2, Ppz, oldPpz1, oldPpz2, Pax, Paz = (zeros(1) for i=1:8)
    else
        isplasma = true
        (; ionrate, rho0, nuc, frequency, Uiev, mr) = plasma

        # Field to intensity conversion coefficient (I = ksi*|E|^2)
        if isdispersion
            chi0 = susceptibility(chi, frequency)
        else
            chi0 = 0
        end
        n = sqrt((eps + real(chi0)) * mu)   # refractive index
        ksi = n * EPS0 * C0 / 2   # 1/2 from <cos^2(t)>

        Ap, Bp, Cp = ade_plasma_coefficients(nuc, mr, dt)
        Ui = Uiev * QE   # eV -> J
        Wph = HBAR * frequency   # energy of one photon
        K = ceil(Ui / Wph)   # minimum number of photons to extract one electron
        sigmaB = QE^2 / (ME*mr) * nuc / (nuc^2 + frequency^2)
        Rava = sigmaB / Ui
        Ma = K * Wph * dt   # update coefficient
        rho, drho = (zeros(Nx,Nz) for i=1:2)
        Ppx, oldPpx1, oldPpx2, Ppz, oldPpz1, oldPpz2, Pax, Paz = (zeros(Nx,Nz) for i=1:8)
    end

    return MaterialStruct2D(
        geometry, sigma, Mh, Me, Md1, Md2, isdispersion, Aq, Bq, Cq, Px, oldPx1, oldPx2, Pz,
        oldPz1, oldPz2, iskerr, Mk2, Mk3, isplasma, ionrate, Rava, ksi, rho0, rho, drho, Ap,
        Bp, Cp, Ppx, oldPpx1, oldPpx2, Ppz, oldPpz1, oldPpz2, Ma, Pax, Paz,
    )
end


# ------------------------------------------------------------------------------------------
struct MaterialStruct3D{G, T, V, A, B, F}
    geometry :: G
    sigma :: T   # conductivity
    # Update coefficients for H, E and D fields:
    Mh :: T
    Me :: T
    Md1 :: T
    Md2 :: T
    # Linear dispersion:
    isdispersion :: Bool
    Aq :: V
    Bq :: V
    Cq :: V
    Px :: A
    oldPx1 :: A
    oldPx2 :: A
    Py :: A
    oldPy1 :: A
    oldPy2 :: A
    Pz :: A
    oldPz1 :: A
    oldPz2 :: A
    # Kerr:
    iskerr :: Bool
    Mk2 :: T
    Mk3 :: T
    # Plasma:
    isplasma :: Bool
    ionrate :: F
    Rava :: T
    ksi :: T   # field to intensity conversion coefficient
    rho0 :: T
    rho :: B   # electron density
    drho :: B   # time derivative of electron density
    Ap :: T
    Bp :: T
    Cp :: T
    Ppx :: B
    oldPpx1 :: B
    oldPpx2 :: B
    Ppy :: B
    oldPpy1 :: B
    oldPpy2 :: B
    Ppz :: B
    oldPpz1 :: B
    oldPpz2 :: B
    Ma :: T
    Pax :: B
    Pay :: B
    Paz :: B
end

@adapt_structure MaterialStruct3D


function MaterialStruct(material, grid::Grid3D, dt)
    if isnothing(material)
        material = Material(geometry = (x,y,z) -> false)
    end
    (; geometry, eps, mu, sigma, chi, chi2, chi3, plasma) = material
    (; Nx, Ny, Nz, x, y, z) = grid

    if typeof(geometry) <: Function
        geometry = [Bool(geometry(x[ix],y[iy],z[iz])) for ix=1:Nx, iy=1:Ny, iz=1:Nz]
    else
        geometry = [Bool(geometry[ix,iy,iz]) for ix=1:Nx, iy=1:Ny, iz=1:Nz]
    end

    sigma = Float64(sigma)
    sigmaD = sigma / (EPS0*eps)   # J=sigma*E -> J=sigma*D

    # Update coefficients for H, E and D fields:
    Mh = dt / (MU0*mu)
    Me = 1 / (EPS0*eps)
    Md1 = (1 - sigmaD*dt/2) / (1 + sigmaD*dt/2)
    Md2 = dt / (1 + sigmaD*dt/2)

    # Variables for ADE dispersion calculation:
    if isnothing(chi)
        isdispersion = false

        # to avoid issues with CUDA kernels, the variables should have the same type for all
        # logical branches:
        Aq, Bq, Cq = (zeros(1) for i=1:3)
        Px, oldPx1, oldPx2 = (zeros(1) for i=1:3)
        Py, oldPy1, oldPy2 = (zeros(1) for i=1:3)
        Pz, oldPz1, oldPz2 = (zeros(1) for i=1:3)
    else
        isdispersion = true

        if chi isa Susceptibility
            chi = (chi,)
        end

        Nq = length(chi)
        Aq, Bq, Cq = (zeros(Nq) for i=1:3)
        for iq=1:Nq
            Aq[iq], Bq[iq], Cq[iq] = ade_coefficients(chi[iq], dt)
        end
        Px, oldPx1, oldPx2 = (zeros(Nq,Nx,Ny,Nz) for i=1:3)
        Py, oldPy1, oldPy2 = (zeros(Nq,Nx,Ny,Nz) for i=1:3)
        Pz, oldPz1, oldPz2 = (zeros(Nq,Nx,Ny,Nz) for i=1:3)
    end

    # Kerr:
    if isnothing(chi2) && isnothing(chi3)
        iskerr = false
        Mk2, Mk3 = 0.0, 0.0
    else
        iskerr = true
        isnothing(chi2) ? Mk2 = 0.0 : Mk2 = EPS0*chi2
        isnothing(chi3) ? Mk3 = 0.0 : Mk3 = EPS0*chi3
    end

    # Plasma:
    if isnothing(plasma)
        isplasma = false
        ionrate = identity
        Rava, ksi, rho0, Ap, Bp, Cp, Ma = (0.0 for i=1:7)
        rho, drho = (zeros(1) for i=1:2)
        Ppx, oldPpx1, oldPpx2 = (zeros(1) for i=1:3)
        Ppy, oldPpy1, oldPpy2 = (zeros(1) for i=1:3)
        Ppz, oldPpz1, oldPpz2 = (zeros(1) for i=1:3)
        Pax, Pay, Paz = (zeros(1) for i=1:3)
    else
        isplasma = true
        (; rho0, ionrate, nuc, frequency, Uiev, mr) = plasma

        # Field to intensity conversion coefficient (I = ksi*|E|^2)
        if isdispersion
            chi0 = susceptibility(chi, frequency)
        else
            chi0 = 0
        end
        n = sqrt((eps + real(chi0)) * mu)   # refractive index
        ksi = n * EPS0 * C0 / 2   # 1/2 from <cos^2(t)>


        Ap, Bp, Cp = ade_plasma_coefficients(nuc, mr, dt)
        rho, drho = (zeros(Nx,Ny,Nz) for i=1:2)
        Ui = Uiev * QE   # eV -> J
        Wph = HBAR * frequency   # energy of one photon
        K = ceil(Ui / Wph)   # minimum number of photons to extract one electron
        sigmaB = QE^2 / (ME*mr) * nuc / (nuc^2 + frequency^2)
        Rava = sigmaB / Ui
        Ma = K * Wph * dt   # update coefficient
        Ppx, oldPpx1, oldPpx2 = (zeros(Nx,Ny,Nz) for i=1:3)
        Ppy, oldPpy1, oldPpy2 = (zeros(Nx,Ny,Nz) for i=1:3)
        Ppz, oldPpz1, oldPpz2 = (zeros(Nx,Ny,Nz) for i=1:3)
        Pax, Pay, Paz = (zeros(Nx,Ny,Nz) for i=1:3)
    end

    return MaterialStruct3D(
        geometry, sigma, Mh, Me, Md1, Md2, isdispersion, Aq, Bq, Cq, Px, oldPx1, oldPx2, Py,
        oldPy1, oldPy2, Pz, oldPz1, oldPz2, iskerr, Mk2, Mk3, isplasma, ionrate, Rava, ksi,
        rho0, rho, drho, Ap, Bp, Cp, Ppx, oldPpx1, oldPpx2, Ppy, oldPpy1, oldPpy2, Ppz,
        oldPpz1, oldPpz2, Ma, Pax, Pay, Paz,
    )
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


function susceptibility(chi::DebyeSusceptibility, w)
    (; deps, tau) = chi
    return deps / (1 - 1im * w * tau)
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


function susceptibility(chi::DrudeSusceptibility, w)
    (; wp, gamma) = chi
    return -wp^2 / (w^2 + 1im * w * gamma)
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


function susceptibility(chi::LorentzSusceptibility, w)
    (; deps, w0, delta) = chi
    return deps * w0^2 / (w0^2 - w^2 - 2im * w * delta)
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


# ------------------------------------------------------------------------------------------
function susceptibility(chis::Union{Vector,Tuple}, w)
    chitot = zero(w)
    for chi in chis
        chitot += susceptibility(chi, w)
    end
    return chitot
end
