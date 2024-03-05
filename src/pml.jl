struct CPML{T}
    thickness :: T
    kmax :: T
    alpha :: T
    R0 :: T
    m :: Int
end


"""
CPML(; thickness, kmax=1, alpha=10e-6, R0=10e-6, m=3)

Convolutional perfectly matched layer with the stretching parameter

    s = kappa + sigma / (alpha - 1im*omega),

where

    kappa = 1 + (kmax-1) (x/L)^m,    sigma = sigma_max * (x/L)^m

with

    sigma_max = -(m+1)*log(R0) / (2*eta0*L).

Here x is the coordinate along the given direction, L is thickness of the PML layer, R0 is
the theoretical reflection coefficient of the PML layer at normal incidence, and eta0 is the
impedance of free space.

# Keywords
- `thickness::Real`: Thickness L of the PML layer.
- `kmax::Real=1`: maximum value of kappa parameter.
- `alpha::Real=10e-6`: alpha parameter.
- `R0::Real=10e-6`: The theoretical reflection coefficient R0 of the PML layer at normal
    incidence
- `m::Int=3`: The power of the losses profile.
"""
function CPML(; thickness, kmax=1, alpha=10e-6, R0=10e-6, m=3)
    thickness, kmax, alpha, R0 = promote(thickness, kmax, alpha, R0)
    return CPML(thickness, kmax, alpha, R0, m)
end


# ******************************************************************************************
# PML layer
# ******************************************************************************************
struct PMLLayer{A}
    ib :: Int   # index of the PML boundary
    Nb :: Int   # total number of grid points in the PML layer
    K :: A
    A :: A
    B :: A
end

@adapt_structure PMLLayer


function LeftPMLLayer(x, Lx, dt; kmax=1, alpha=10e-6, R0=10e-6, m=3)
    if Lx == 0
        ixb = 1
        K = ones(1)
        B = zeros(1)
        A = zeros(1)
    else
        xb = x[1] + Lx
        ixb = argmin(abs.(x .- xb))
        Nxb = ixb

        eta0 = sqrt(MU0 / EPS0)
        smax = -(m + 1) * log(R0) / (2 * eta0 * Lx)

        kappa = zeros(Nxb)
        sigma = zeros(Nxb)
        for ix=1:Nxb
            ixpml = ix
            kappa[ixpml] = 1 + (kmax - 1) * (abs(x[ix] - xb) / Lx)^m
            sigma[ixpml] = smax * (abs(x[ix] - xb) / Lx)^m
        end

        K = kappa
        B = @. exp(-(sigma / K + alpha) * dt / EPS0)
        A = @. sigma / (sigma * K + alpha * K^2) * (B - 1)
    end
    return PMLLayer(ixb, Nxb, K, A, B)
end


function LeftPMLLayer(x, Lx::CPML, dt)
    (; thickness, kmax, alpha, R0, m) = Lx
    return LeftPMLLayer(x, thickness, dt; kmax, alpha, R0, m)
end


function RightPMLLayer(x, Lx, dt; kmax=1, alpha=10e-6, R0=10e-6, m=3)
    if Lx == 0
        ixb = length(x)
        K = ones(1)
        B = zeros(1)
        A = zeros(1)
    else
        Nx = length(x)
        xb = x[end] - Lx
        ixb = argmin(abs.(x .- xb))
        Nxb = Nx - ixb + 1

        eta0 = sqrt(MU0 / EPS0)
        smax = -(m + 1) * log(R0) / (2 * eta0 * Lx)

        kappa = zeros(Nxb)
        sigma = zeros(Nxb)
        for ix=ixb:Nx
            ixpml = ix - ixb + 1
            kappa[ixpml] = 1 + (kmax - 1) * (abs(x[ix] - xb) / Lx)^m
            sigma[ixpml] = smax * (abs(x[ix] - xb) / Lx)^m
        end

        K = kappa
        B = @. exp(-(sigma / K + alpha) * dt / EPS0)
        A = @. sigma / (sigma * K + alpha * K^2) * (B - 1)
    end
    return PMLLayer(ixb, Nxb, K, A, B)
end


function RightPMLLayer(x, Lx::CPML, dt)
    (; thickness, kmax, alpha, R0, m) = Lx
    return RightPMLLayer(x, thickness, dt; kmax, alpha, R0, m)
end


# ******************************************************************************************
# 1D
# ******************************************************************************************
struct PML1D{L, A}
    # z left layer:
    zlayer1 :: L
    psiHyz1 :: A
    psiExz1 :: A
    # z right layer:
    zlayer2 :: L
    psiHyz2 :: A
    psiExz2 :: A
end

@adapt_structure PML1D


function PML(grid::Grid1D, pml, dt)
    (; z) = grid

    if pml isa Real || pml isa CPML
        Lz1 = Lz2 = pml
    else
        Lz1, Lz2 = pml
    end

    zlayer1 = LeftPMLLayer(z, Lz1, dt)
    psiHyz1, psiExz1 = zeros(zlayer1.Nb), zeros(zlayer1.Nb)

    zlayer2 = RightPMLLayer(z, Lz2, dt)
    psiHyz2, psiExz2 = zeros(zlayer2.Nb), zeros(zlayer2.Nb)

    return PML1D(
        zlayer1, psiHyz1, psiExz1,
        zlayer2, psiHyz2, psiExz2,
    )
end


# ******************************************************************************************
# 2D
# ******************************************************************************************
struct PML2D{L, A}
    # x left layer:
    xlayer1 :: L
    psiHyx1 :: A
    psiEzx1 :: A
    # x right layer:
    xlayer2 :: L
    psiHyx2 :: A
    psiEzx2 :: A
    # z left layer:
    zlayer1 :: L
    psiHyz1 :: A
    psiExz1 :: A
    # z right layer:
    zlayer2 :: L
    psiHyz2 :: A
    psiExz2 :: A
end

@adapt_structure PML2D


function PML(grid::Grid2D, pml, dt)
    (; Nx, Nz, x, z) = grid

    if pml isa Real || pml isa CPML
        Lx1 = Lx2 = Lz1 = Lz2 = pml
    else
        Lx1, Lx2, Lz1, Lz2 = pml
    end

    xlayer1 = LeftPMLLayer(x, Lx1, dt)
    psiHyx1, psiEzx1 = zeros(xlayer1.Nb,Nz), zeros(xlayer1.Nb,Nz)

    xlayer2 = RightPMLLayer(x, Lx2, dt)
    psiHyx2, psiEzx2 = zeros(xlayer2.Nb,Nz), zeros(xlayer2.Nb,Nz)

    zlayer1 = LeftPMLLayer(z, Lz1, dt)
    psiHyz1, psiExz1 = zeros(Nx,zlayer1.Nb), zeros(Nx,zlayer1.Nb)

    zlayer2 = RightPMLLayer(z, Lz2, dt)
    psiHyz2, psiExz2 = zeros(Nx,zlayer2.Nb), zeros(Nx,zlayer2.Nb)

    return PML2D(
        xlayer1, psiHyx1, psiEzx1,
        xlayer2, psiHyx2, psiEzx2,
        zlayer1, psiHyz1, psiExz1,
        zlayer2, psiHyz2, psiExz2,
    )
end


# ******************************************************************************************
# 3D
# ******************************************************************************************
struct PML3D{L, A}
    # x left layer:
    xlayer1 :: L
    psiHyx1 :: A
    psiHzx1 :: A
    psiEyx1 :: A
    psiEzx1 :: A
    # x right layer:
    xlayer2 :: L
    psiHyx2 :: A
    psiHzx2 :: A
    psiEyx2 :: A
    psiEzx2 :: A
    # y left layer:
    ylayer1 :: L
    psiHxy1 :: A
    psiHzy1 :: A
    psiExy1 :: A
    psiEzy1 :: A
    # y right layer:
    ylayer2 :: L
    psiHxy2 :: A
    psiHzy2 :: A
    psiExy2 :: A
    psiEzy2 :: A
    # z left layer:
    zlayer1 :: L
    psiHxz1 :: A
    psiHyz1 :: A
    psiExz1 :: A
    psiEyz1 :: A
    # z right layer:
    zlayer2 :: L
    psiHxz2 :: A
    psiHyz2 :: A
    psiExz2 :: A
    psiEyz2 :: A
end

@adapt_structure PML3D


function PML(grid::Grid3D, pml, dt)
    (; Nx, Ny, Nz, x, y, z) = grid

    if pml isa Real || pml isa CPML
        Lx1 = Lx2 = Ly1 = Ly2 = Lz1 = Lz2 = pml
    else
        Lx1, Lx2, Ly1, Ly2, Lz1, Lz2 = pml
    end

    xlayer1 = LeftPMLLayer(x, Lx1, dt)
    psiHyx1, psiHzx1, psiEyx1, psiEzx1 = (zeros(xlayer1.Nb,Ny,Nz) for i=1:4)

    xlayer2 = RightPMLLayer(x, Lx2, dt)
    psiHyx2, psiHzx2, psiEyx2, psiEzx2 = (zeros(xlayer2.Nb,Ny,Nz) for i=1:4)

    ylayer1 = LeftPMLLayer(y, Ly1, dt)
    psiHxy1, psiHzy1, psiExy1, psiEzy1 = (zeros(Nx,ylayer1.Nb,Nz) for i=1:4)

    ylayer2 = RightPMLLayer(y, Ly2, dt)
    psiHxy2, psiHzy2, psiExy2, psiEzy2 = (zeros(Nx,ylayer2.Nb,Nz) for i=1:4)

    zlayer1 = LeftPMLLayer(z, Lz1, dt)
    psiHxz1, psiHyz1, psiExz1, psiEyz1 = (zeros(Nx,Ny,zlayer1.Nb) for i=1:4)

    zlayer2 = RightPMLLayer(z, Lz2, dt)
    psiHxz2, psiHyz2, psiExz2, psiEyz2 = (zeros(Nx,Ny,zlayer2.Nb) for i=1:4)

    return PML3D(
        xlayer1, psiHyx1, psiHzx1, psiEyx1, psiEzx1,
        xlayer2, psiHyx2, psiHzx2, psiEyx2, psiEzx2,
        ylayer1, psiHxy1, psiHzy1, psiExy1, psiEzy1,
        ylayer2, psiHxy2, psiHzy2, psiExy2, psiEzy2,
        zlayer1, psiHxz1, psiHyz1, psiExz1, psiEyz1,
        zlayer2, psiHxz2, psiHyz2, psiExz2, psiEyz2,
    )
end
