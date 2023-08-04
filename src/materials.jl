struct MaterialData{G, T, C, C3}
    geometry :: G
    eps :: T
    mu :: T
    sigma :: T
    chi :: C
    chi3 :: C3
end


function Material(; geometry, eps=1, mu=1, sigma=0, chi=nothing, chi3=nothing)
    eps, mu, sigma = promote(eps, mu, sigma)
    return MaterialData(geometry, eps, mu, sigma, chi, chi3)
end


# ******************************************************************************************
# Materials
# ******************************************************************************************
struct Material1D{A, B}
    # Linear dispersion:
    dispersion :: Bool
    Aq :: A
    Bq :: A
    Cq :: A
    Px :: A
    oldPx1 :: A
    oldPx2 :: A
    # Kerr:
    kerr :: Bool
    Mk :: B
    # Plasma:
    # rho :: B
    # drho :: B
    # Ppx :: B
    # oldPpx1 :: B
    # oldPpx2 :: B
    # Pax :: B
end

@adapt_structure Material1D


function material_init(material_data, grid::Grid1D, dt)
    if isnothing(material_data)
        material_data = Material(geometry = z -> false)
    end
    (; geometry, eps, mu, sigma, chi, chi3) = material_data
    (; Nz, z) = grid

    # Permittivity, permeability, and conductivity:
    eps = [geometry(z[iz]) ? eps : 1 for iz=1:Nz]
    mu = [geometry(z[iz]) ? mu : 1 for iz=1:Nz]
    sigma = [geometry(z[iz]) ? sigma : 0 for iz=1:Nz]
    @. sigma = sigma / (EPS0*eps)   # J=sigma*E -> J=sigma*D

    # Variables for ADE dispersion calculation:
    if isnothing(chi)
        dispersion = false

        # to avoid issues with CUDA kernel we use zeros(1) instead of nothing
        Aq, Bq, Cq = (zeros(1) for i=1:3)
        Px, oldPx1, oldPx2 = (zeros(1) for i=1:3)
    else
        dispersion = true

        if chi isa Susceptibility
            chi = (chi,)
        end

        Nq = length(chi)
        Aq, Bq, Cq = (zeros(Nq,Nz) for i=1:3)
        for iz=1:Nz, iq=1:Nq
            Aq0, Bq0, Cq0 = ade_coefficients(chi[iq], dt)
            Aq[iq,iz] = geometry(z[iz]) * Aq0
            Bq[iq,iz] = geometry(z[iz]) * Bq0
            Cq[iq,iz] = geometry(z[iz]) * Cq0
        end
        Px, oldPx1, oldPx2 = (zeros(Nq,Nz) for i=1:3)
    end

    # Kerr:
    if isnothing(chi3)
        kerr = false
        Mk = zeros(1)
    else
        kerr = true
        Mk = [geometry(z[iz]) ? EPS0*chi3 : 0 for iz=1:Nz]
    end

    # rho = zeros(Nz)   # electron density
    # drho = zeros(Nz)   # derivative of electron density
    # Ppx, oldPpx1, oldPpx2 = (zeros(Nz) for i=1:3)   # plasma current
    # Pax = zeros(Nz)   # multiphoton losses

    material = Material1D(
        dispersion, Aq, Bq, Cq, Px, oldPx1, oldPx2,
        kerr, Mk,
        # rho, drho, Ppx, oldPpx1, oldPpx2, Pax,
    )
    return eps, mu, sigma, material
end


# ------------------------------------------------------------------------------------------
struct Material2D{A, B}
    # Linear dispersion:
    dispersion :: Bool
    Aq :: A
    Bq :: A
    Cq :: A
    Px :: A
    oldPx1 :: A
    oldPx2 :: A
    Pz :: A
    oldPz1 :: A
    oldPz2 :: A
    # Kerr:
    kerr :: Bool
    Mk :: B
end

@adapt_structure Material2D


function material_init(material_data, grid::Grid2D, dt)
    if isnothing(material_data)
        material_data = Material(geometry = (x,z) -> false)
    end
    (; geometry, eps, mu, sigma, chi, chi3) = material_data
    (; Nx, Nz, x, z) = grid

    # Permittivity, permeability, and conductivity:
    eps = [geometry(x[ix],z[iz]) ? eps : 1 for ix=1:Nx, iz=1:Nz]
    mu = [geometry(x[ix],z[iz]) ? mu : 1 for ix=1:Nx, iz=1:Nz]
    sigma = [geometry(x[ix],z[iz]) ? sigma : 0 for ix=1:Nx, iz=1:Nz]
    @. sigma = sigma / (EPS0*eps)   # J=sigma*E -> J=sigma*D

    # Variables for ADE dispersion calculation:
    if isnothing(chi)
        dispersion = false

        # to avoid issues with CUDA kernel we use zeros(1) instead of nothing
        Aq, Bq, Cq = (zeros(1) for i=1:3)
        Px, oldPx1, oldPx2 = (zeros(1) for i=1:3)
        Pz, oldPz1, oldPz2 = (zeros(1) for i=1:3)
    else
        dispersion = true

        if chi isa Susceptibility
            chi = (chi,)
        end

        Nq = length(chi)
        Aq, Bq, Cq = (zeros(Nq,Nx,Nz) for i=1:3)
        for iz=1:Nz, ix=1:Nx, iq=1:Nq
            Aq0, Bq0, Cq0 = ade_coefficients(chi[iq], dt)
            Aq[iq,ix,iz] = geometry(x[ix],z[iz]) * Aq0
            Bq[iq,ix,iz] = geometry(x[ix],z[iz]) * Bq0
            Cq[iq,ix,iz] = geometry(x[ix],z[iz]) * Cq0
        end
        Px, oldPx1, oldPx2 = (zeros(Nq,Nx,Nz) for i=1:3)
        Pz, oldPz1, oldPz2 = (zeros(Nq,Nx,Nz) for i=1:3)
    end

    # Kerr:
    if isnothing(chi3)
        kerr = false
        Mk = zeros(1)
    else
        kerr = true
        Mk = [geometry(x[ix],z[iz]) ? EPS0*chi3 : 0 for ix=1:Nx, iz=1:Nz]
    end

    material = Material2D(
        dispersion, Aq, Bq, Cq, Px, oldPx1, oldPx2, Pz, oldPz1, oldPz2,
        kerr, Mk,
    )
    return eps, mu, sigma, material
end


# ------------------------------------------------------------------------------------------
struct Material3D{A, B}
    # Linear dispersion:
    dispersion :: Bool
    Aq :: A
    Bq :: A
    Cq :: A
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
    kerr :: Bool
    Mk :: B
end

@adapt_structure Material3D


function material_init(material_data, grid::Grid3D, dt)
    if isnothing(material_data)
        material_data = Material(geometry = (x,y,z) -> false)
    end
    (; geometry, eps, mu, sigma, chi, chi3) = material_data
    (; Nx, Ny, Nz, x, y, z) = grid

    # Permittivity, permeability, and conductivity:
    eps = [geometry(x[ix],y[iy],z[iz]) ? eps : 1 for ix=1:Nx, iy=1:Ny, iz=1:Nz]
    mu = [geometry(x[ix],y[iy],z[iz]) ? mu : 1 for ix=1:Nx, iy=1:Ny, iz=1:Nz]
    sigma = [geometry(x[ix],y[iy],z[iz]) ? sigma : 0 for ix=1:Nx, iy=1:Ny, iz=1:Nz]
    @. sigma = sigma / (EPS0*eps)   # J=sigma*E -> J=sigma*D

    # Variables for ADE dispersion calculation:
    if isnothing(chi)
        dispersion = false

        # to avoid issues with CUDA kernel we use zeros(1) instead of nothing
        Aq, Bq, Cq = (zeros(1) for i=1:3)
        Px, oldPx1, oldPx2 = (zeros(1) for i=1:3)
        Py, oldPy1, oldPy2 = (zeros(1) for i=1:3)
        Pz, oldPz1, oldPz2 = (zeros(1) for i=1:3)
    else
        dispersion = true

        if chi isa Susceptibility
            chi = (chi,)
        end

        Nq = length(chi)
        Aq, Bq, Cq = (zeros(Nq,Nx,Ny,Nz) for i=1:3)
        for iz=1:Nz, iy=1:Ny, ix=1:Nx, iq=1:Nq
            Aq0, Bq0, Cq0 = ade_coefficients(chi[iq], dt)
            Aq[iq,ix,iy,iz] = geometry(x[ix],y[iy],z[iz]) * Aq0
            Bq[iq,ix,iy,iz] = geometry(x[ix],y[iy],z[iz]) * Bq0
            Cq[iq,ix,iy,iz] = geometry(x[ix],y[iy],z[iz]) * Cq0
        end
        Px, oldPx1, oldPx2 = (zeros(Nq,Nx,Ny,Nz) for i=1:3)
        Py, oldPy1, oldPy2 = (zeros(Nq,Nx,Ny,Nz) for i=1:3)
        Pz, oldPz1, oldPz2 = (zeros(Nq,Nx,Ny,Nz) for i=1:3)
    end

    # Kerr:
    if isnothing(chi3)
        kerr = false
        Mk = zeros(1)
    else
        kerr = true
        Mk = [geometry(x[ix],y[iy],z[iz]) ? EPS0*chi3 : 0 for ix=1:Nx, iy=1:Ny, iz=1:Nz]
    end

    material = Material3D(
        dispersion, Aq, Bq, Cq, Px, oldPx1, oldPx2, Py, oldPy1, oldPy2, Pz, oldPz1, oldPz2,
        kerr, Mk,
    )
    return eps, mu, sigma, material
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
