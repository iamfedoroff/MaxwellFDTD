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
    ind :: Int
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
        Nxpml = ixb

        eta0 = sqrt(MU0 / EPS0)
        sigma_max = -(m + 1) * log(R0) / (2 * eta0 * Lx)

        sigma = zeros(Nxpml)
        for ix=1:Nxpml
            ixpml = ix
            sigma[ixpml] = sigma_max * (abs(x[ix] - xb) / Lx)^m
        end

        kappa = zeros(Nxpml)
        for ix=1:Nxpml
            ixpml = ix
            kappa[ixpml] = 1 + (kmax - 1) * (abs(x[ix] - xb) / Lx)^m
        end

        K = kappa
        B = @. exp(-(sigma / K + alpha) * dt / EPS0)
        A = @. sigma / (sigma * K + alpha * K^2) * (B - 1)
    end
    return PMLLayer(ixb, K, A, B)
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
        Nxpml = Nx - ixb + 1

        eta0 = sqrt(MU0 / EPS0)
        sigma_max = -(m + 1) * log(R0) / (2 * eta0 * Lx)

        sigma = zeros(Nxpml)
        for ix=ixb:Nx
            ixpml = ix - ixb + 1
            sigma[ixpml] = sigma_max * (abs(x[ix] - xb) / Lx)^m
        end

        kappa = zeros(Nxpml)
        for ix=ixb:Nx
            ixpml = ix - ixb + 1
            kappa[ixpml] = 1 + (kmax - 1) * (abs(x[ix] - xb) / Lx)^m
        end

        K = kappa
        B = @. exp(-(sigma / K + alpha) * dt / EPS0)
        A = @. sigma / (sigma * K + alpha * K^2) * (B - 1)
    end
    return PMLLayer(ixb, K, A, B)
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
    (; Nz, z) = grid

    if pml isa Real || pml isa CPML
        Lz1 = Lz2 = pml
    else
        Lz1, Lz2 = pml
    end

    if Lz1 isa CPML
        (; thickness, kmax, alpha, R0, m) = Lz1
        zlayer1 = LeftPMLLayer(z, thickness, dt; kmax, alpha, R0, m)
    else
        zlayer1 = LeftPMLLayer(z, Lz1, dt)
    end
    Nzpml = zlayer1.ind
    psiHyz1, psiExz1 = zeros(Nzpml), zeros(Nzpml)

    if Lz2 isa CPML
        (; thickness, kmax, alpha, R0, m) = Lz2
        zlayer2 = RightPMLLayer(z, thickness, dt; kmax, alpha, R0, m)
    else
        zlayer2 = RightPMLLayer(z, Lz2, dt)
    end
    Nzpml = Nz - zlayer2.ind + 1
    psiHyz2, psiExz2 = zeros(Nzpml), zeros(Nzpml)

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

    if Lx1 isa CPML
        (; thickness, kmax, alpha, R0, m) = Lx1
        xlayer1 = LeftPMLLayer(x, thickness, dt; kmax, alpha, R0, m)
    else
        xlayer1 = LeftPMLLayer(x, Lx1, dt)
    end
    Nxpml = xlayer1.ind
    psiHyx1, psiEzx1 = zeros(Nxpml,Nz), zeros(Nxpml,Nz)

    if Lx2 isa CPML
        (; thickness, kmax, alpha, R0, m) = Lx2
        xlayer2 = RightPMLLayer(x, thickness, dt; kmax, alpha, R0, m)
    else
        xlayer2 = RightPMLLayer(x, Lx2, dt)
    end
    Nxpml = Nx - xlayer2.ind + 1
    psiHyx2, psiEzx2 = zeros(Nxpml,Nz), zeros(Nxpml,Nz)

    if Lz1 isa CPML
        (; thickness, kmax, alpha, R0, m) = Lz1
        zlayer1 = LeftPMLLayer(z, thickness, dt; kmax, alpha, R0, m)
    else
        zlayer1 = LeftPMLLayer(z, Lz1, dt)
    end
    Nzpml = zlayer1.ind
    psiHyz1, psiExz1 = zeros(Nx,Nzpml), zeros(Nx,Nzpml)

    if Lz2 isa CPML
        (; thickness, kmax, alpha, R0, m) = Lz2
        zlayer2 = RightPMLLayer(z, thickness, dt; kmax, alpha, R0, m)
    else
        zlayer2 = RightPMLLayer(z, Lz2, dt)
    end
    Nzpml = Nz - zlayer2.ind + 1
    psiHyz2, psiExz2 = zeros(Nx,Nzpml), zeros(Nx,Nzpml)

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

    if Lx1 isa CPML
        (; thickness, kmax, alpha, R0, m) = Lx1
        xlayer1 = LeftPMLLayer(x, thickness, dt; kmax, alpha, R0, m)
    else
        xlayer1 = LeftPMLLayer(x, Lx1, dt)
    end
    Nxpml = xlayer1.ind
    psiHyx1, psiHzx1, psiEyx1, psiEzx1 = (zeros(Nxpml,Ny,Nz) for i=1:4)

    if Lx2 isa CPML
        (; thickness, kmax, alpha, R0, m) = Lx2
        xlayer2 = RightPMLLayer(x, thickness, dt; kmax, alpha, R0, m)
    else
        xlayer2 = RightPMLLayer(x, Lx2, dt)
    end
    Nxpml = Nx - xlayer2.ind + 1
    psiHyx2, psiHzx2, psiEyx2, psiEzx2 = (zeros(Nxpml,Ny,Nz) for i=1:4)

    if Ly1 isa CPML
        (; thickness, kmax, alpha, R0, m) = Ly1
        ylayer1 = LeftPMLLayer(y, thickness, dt; kmax, alpha, R0, m)
    else
        ylayer1 = LeftPMLLayer(y, Ly1, dt)
    end
    Nypml = ylayer1.ind
    psiHxy1, psiHzy1, psiExy1, psiEzy1 = (zeros(Nx,Nypml,Nz) for i=1:4)

    if Ly2 isa CPML
        (; thickness, kmax, alpha, R0, m) = Ly2
        ylayer2 = RightPMLLayer(y, thickness, dt; kmax, alpha, R0, m)
    else
        ylayer2 = RightPMLLayer(y, Ly2, dt)
    end
    Nypml = Ny - ylayer2.ind + 1
    psiHxy2, psiHzy2, psiExy2, psiEzy2 = (zeros(Nx,Nypml,Nz) for i=1:4)

    if Lz1 isa CPML
        (; thickness, kmax, alpha, R0, m) = Lz1
        zlayer1 = LeftPMLLayer(z, thickness, dt; kmax, alpha, R0, m)
    else
        zlayer1 = LeftPMLLayer(z, Lz1, dt)
    end
    Nzpml = zlayer1.ind
    psiHxz1, psiHyz1, psiExz1, psiEyz1 = (zeros(Nx,Ny,Nzpml) for i=1:4)

    if Lz2 isa CPML
        (; thickness, kmax, alpha, R0, m) = Lz2
        zlayer2 = RightPMLLayer(z, thickness, dt; kmax, alpha, R0, m)
    else
        zlayer2 = RightPMLLayer(z, Lz2, dt)
    end
    Nzpml = Nz - zlayer2.ind + 1
    psiHxz2, psiHyz2, psiExz2, psiEyz2 = (zeros(Nx,Ny,Nzpml) for i=1:4)

    return PML3D(
        xlayer1, psiHyx1, psiHzx1, psiEyx1, psiEzx1,
        xlayer2, psiHyx2, psiHzx2, psiEyx2, psiEzx2,
        ylayer1, psiHxy1, psiHzy1, psiExy1, psiEzy1,
        ylayer2, psiHxy2, psiHzy2, psiExy2, psiEzy2,
        zlayer1, psiHxz1, psiHyz1, psiExz1, psiEyz1,
        zlayer2, psiHxz2, psiHyz2, psiExz2, psiEyz2,
    )
end
