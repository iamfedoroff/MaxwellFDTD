struct PMLData{T}
    thickness :: T
    kappa :: T
    alpha :: T
    R0 :: T
    m :: Int
end


function CPML(; thickness, kappa=1, alpha=10e-6, R0=10e-6, m=3)
    thickness, kappa, alpha, R0 = promote(thickness, kappa, alpha, R0)
    return PMLData(thickness, kappa, alpha, R0, m)
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


function LeftPMLLayer(x, Lx, dt; kappa=1, alpha=10e-6, R0=10e-6, m=3)
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

        K = ones(Nxpml) * kappa
        B = @. exp(-(sigma / K + alpha) * dt / EPS0)
        A = @. sigma / (sigma * K + alpha * K^2) * (B - 1)
    end
    return PMLLayer(ixb, K, A, B)
end


function RightPMLLayer(x, Lx, dt; kappa=1, alpha=10e-6, R0=10e-6, m=3)
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

        K = ones(Nxpml) * kappa
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


function PML(grid::Grid1D, box, dt)
    (; Nz, z) = grid

    if isnothing(box)
        Lz1, Lz2 = 0, 0
    else
        Lz1, Lz2 = box
    end

    if typeof(Lz1) <: PMLData
        (; thickness, kappa, alpha, R0, m) = Lz1
        zlayer1 = LeftPMLLayer(z, thickness, dt; kappa, alpha, R0, m)
    else
        zlayer1 = LeftPMLLayer(z, Lz1, dt)
    end
    Nzpml = zlayer1.ind
    psiHyz1, psiExz1 = zeros(Nzpml), zeros(Nzpml)

    if typeof(Lz2) <: PMLData
        (; thickness, kappa, alpha, R0, m) = Lz2
        zlayer2 = RightPMLLayer(z, thickness, dt; kappa, alpha, R0, m)
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


function PML(grid::Grid2D, box, dt)
    (; Nx, Nz, x, z) = grid

    if isnothing(box)
        Lx1, Lx2, Lz1, Lz2 = 0, 0, 0, 0
    else
        Lx1, Lx2, Lz1, Lz2 = box
    end

    if typeof(Lx1) <: PMLData
        (; thickness, kappa, alpha, R0, m) = Lx1
        xlayer1 = LeftPMLLayer(x, thickness, dt; kappa, alpha, R0, m)
    else
        xlayer1 = LeftPMLLayer(x, Lx1, dt)
    end
    Nxpml = xlayer1.ind
    psiHyx1, psiEzx1 = zeros(Nxpml,Nz), zeros(Nxpml,Nz)

    if typeof(Lx2) <: PMLData
        (; thickness, kappa, alpha, R0, m) = Lx2
        xlayer2 = RightPMLLayer(x, thickness, dt; kappa, alpha, R0, m)
    else
        xlayer2 = RightPMLLayer(x, Lx2, dt)
    end
    Nxpml = Nx - xlayer2.ind + 1
    psiHyx2, psiEzx2 = zeros(Nxpml,Nz), zeros(Nxpml,Nz)

    if typeof(Lz1) <: PMLData
        (; thickness, kappa, alpha, R0, m) = Lz1
        zlayer1 = LeftPMLLayer(z, thickness, dt; kappa, alpha, R0, m)
    else
        zlayer1 = LeftPMLLayer(z, Lz1, dt)
    end
    Nzpml = zlayer1.ind
    psiHyz1, psiExz1 = zeros(Nx,Nzpml), zeros(Nx,Nzpml)

    if typeof(Lz2) <: PMLData
        (; thickness, kappa, alpha, R0, m) = Lz2
        zlayer2 = RightPMLLayer(z, thickness, dt; kappa, alpha, R0, m)
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


function PML(grid::Grid3D, box, dt)
    (; Nx, Ny, Nz, x, y, z) = grid

    if isnothing(box)
        Lx1, Lx2, Ly1, Ly2, Lz1, Lz2 = 0, 0, 0, 0, 0, 0
    else
        Lx1, Lx2, Ly1, Ly2, Lz1, Lz2 = box
    end

    if typeof(Lx1) <: PMLData
        (; thickness, kappa, alpha, R0, m) = Lx1
        xlayer1 = LeftPMLLayer(x, thickness, dt; kappa, alpha, R0, m)
    else
        xlayer1 = LeftPMLLayer(x, Lx1, dt)
    end
    Nxpml = xlayer1.ind
    psiHyx1, psiHzx1, psiEyx1, psiEzx1 = (zeros(Nxpml,Ny,Nz) for i=1:4)

    if typeof(Lx2) <: PMLData
        (; thickness, kappa, alpha, R0, m) = Lx2
        xlayer2 = RightPMLLayer(x, thickness, dt; kappa, alpha, R0, m)
    else
        xlayer2 = RightPMLLayer(x, Lx2, dt)
    end
    Nxpml = Nx - xlayer2.ind + 1
    psiHyx2, psiHzx2, psiEyx2, psiEzx2 = (zeros(Nxpml,Ny,Nz) for i=1:4)

    if typeof(Ly1) <: PMLData
        (; thickness, kappa, alpha, R0, m) = Ly1
        ylayer1 = LeftPMLLayer(y, thickness, dt; kappa, alpha, R0, m)
    else
        ylayer1 = LeftPMLLayer(y, Ly1, dt)
    end
    Nypml = ylayer1.ind
    psiHxy1, psiHzy1, psiExy1, psiEzy1 = (zeros(Nx,Nypml,Nz) for i=1:4)

    if typeof(Ly2) <: PMLData
        (; thickness, kappa, alpha, R0, m) = Ly2
        ylayer2 = RightPMLLayer(y, thickness, dt; kappa, alpha, R0, m)
    else
        ylayer2 = RightPMLLayer(y, Ly2, dt)
    end
    Nypml = Ny - ylayer2.ind + 1
    psiHxy2, psiHzy2, psiExy2, psiEzy2 = (zeros(Nx,Nypml,Nz) for i=1:4)

    if typeof(Lz1) <: PMLData
        (; thickness, kappa, alpha, R0, m) = Lz1
        zlayer1 = LeftPMLLayer(z, thickness, dt; kappa, alpha, R0, m)
    else
        zlayer1 = LeftPMLLayer(z, Lz1, dt)
    end
    Nzpml = zlayer1.ind
    psiHxz1, psiHyz1, psiExz1, psiEyz1 = (zeros(Nx,Ny,Nzpml) for i=1:4)

    if typeof(Lz2) <: PMLData
        (; thickness, kappa, alpha, R0, m) = Lz2
        zlayer2 = RightPMLLayer(z, thickness, dt; kappa, alpha, R0, m)
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
