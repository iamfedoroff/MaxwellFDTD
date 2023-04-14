function pml(x, box, dt; kappa=1, alpha=10e-6, R0=10e-6, m=3)
    Lx1, Lx2 = box
    Nx = length(x)
    xmin, xmax = x[1], x[end]

    eta0 = sqrt(MU0 / EPS0)
    sigma_max1 = -(m + 1) * log(R0) / (2 * eta0 * Lx1)
    sigma_max2 = -(m + 1) * log(R0) / (2 * eta0 * Lx2)

    sigma = zeros(Nx)
    xb1, xb2 = xmin + Lx1, xmax - Lx2
    for ix=1:Nx
        if x[ix] < xb1
            sigma[ix] = sigma_max1 * (abs(x[ix] - xb1) / Lx1)^m
        end
        if x[ix] > xb2
            sigma[ix] = sigma_max2 * (abs(x[ix] - xb2) / Lx2)^m
        end
    end

    K = ones(Nx) * kappa
    B = @. exp(-(sigma / K + alpha) * dt / EPS0)
    A = @. sigma / (sigma * K + alpha * K^2) * (B - 1)

    return K, A, B
end


# ******************************************************************************
# 2D
# ******************************************************************************
function pml(grid::Grid2D, box)
    (; Nx, Nz, x, z) = grid
    xmin, xmax = x[1], x[end]
    zmin, zmax = z[1], z[end]

    xpml1, xpml2, zpml1, zpml2 = box

    sx = zeros(Nx, Nz)
    sz = zeros(Nx, Nz)

    xb1, xb2 = xmin + xpml1, xmax - xpml2
    zb1, zb2 = zmin + zpml1, zmax - zpml2
    for iz=1:Nz, ix=1:Nx
        if x[ix] < xb1
            sx[ix,iz] = abs((x[ix] - xb1) / xpml1)^3
        end
        if x[ix] > xb2
            sx[ix,iz] = abs((x[ix] - xb2) / xpml2)^3
        end
        if z[iz] < zb1
            sz[ix,iz] = abs((z[iz] - zb1) / zpml1)^3
        end
        if z[iz] > zb2
            sz[ix,iz] = abs((z[iz] - zb2) / zpml2)^3
        end
    end

    return sx, sz
end



# ******************************************************************************
# 3D
# ******************************************************************************
function pml(grid::Grid3D, box)
    (; Nx, Ny, Nz, x, y, z) = grid
    xmin, xmax = x[1], x[end]
    ymin, ymax = y[1], y[end]
    zmin, zmax = z[1], z[end]

    xpml1, xpml2, ypml1, ypml2, zpml1, zpml2 = box

    sx = zeros(Nx, Ny, Nz)
    sy = zeros(Nx, Ny, Nz)
    sz = zeros(Nx, Ny, Nz)

    # box:
    xb1, xb2 = xmin + xpml1, xmax - xpml2
    yb1, yb2 = ymin + ypml1, ymax - ypml2
    zb1, zb2 = zmin + zpml1, zmax - zpml2
    for iz=1:Nz, iy=1:Ny, ix=1:Nx
        if x[ix] < xb1
            sx[ix,iy,iz] = abs((x[ix] - xb1) / xpml1)^3
        end
        if x[ix] > xb2
            sx[ix,iy,iz] = abs((x[ix] - xb2) / xpml2)^3
        end
        if y[iy] < yb1
            sy[ix,iy,iz] = abs((y[iy] - yb1) / ypml1)^3
        end
        if y[iy] > yb2
            sy[ix,iy,iz] = abs((y[iy] - yb2) / ypml2)^3
        end
        if z[iz] < zb1
            sz[ix,iy,iz] = abs((z[iz] - zb1) / zpml1)^3
        end
        if z[iz] > zb2
            sz[ix,iy,iz] = abs((z[iz] - zb2) / zpml2)^3
        end
    end

    # z cylinder:
    # rad = xmax - xpml1
    # rmax = xmax
    # zb1, zb2 = zmin + zpml1, zmax - zpml2
    # for iz=1:Nz, iy=1:Ny, ix=1:Nx
    #     r = sqrt(x[ix]^2 + y[iy]^2)
    #     if r > rad
    #         sx[ix,iy,iz] = abs((r - rad) / (rmax - rad))^3
    #         sy[ix,iy,iz] = abs((r - rad) / (rmax - rad))^3
    #     end
    #     if z[iz] < zb1
    #         sz[ix,iy,iz] = abs((z[iz] - zb1) / zpml1)^3
    #     end
    #     if z[iz] > zb2
    #         sz[ix,iy,iz] = abs((z[iz] - zb2) / zpml2)^3
    #     end
    # end

    return sx, sy, sz
end
