# ******************************************************************************
# 1D
# ******************************************************************************
function pml(grid::Grid1D, box)
    (; Nz, z) = grid
    zmin, zmax = z[1], z[end]

    zpml1, zpml2 = box

    sz = zeros(Nz)

    zb1, zb2 = zmin + zpml1, zmax - zpml2
    for iz=1:Nz
        if z[iz] < zb1
            sz[iz] = abs((z[iz] - zb1) / zpml1)^3
        end
        if z[iz] > zb2
            sz[iz] = abs((z[iz] - zb2) / zpml2)^3
        end
    end

    return sz
end


# ******************************************************************************
# 2D
# ******************************************************************************
function pml(grid::Grid2D, box)
    (; Nx, Ny, x, y) = grid
    xmin, xmax = x[1], x[end]
    ymin, ymax = y[1], y[end]

    xpml1, xpml2, ypml1, ypml2 = box

    sx = zeros(Nx, Ny)
    sy = zeros(Nx, Ny)

    # box:
    xb1, xb2 = xmin + xpml1, xmax - xpml2
    yb1, yb2 = ymin + ypml1, ymax - ypml2
    for iy=1:Ny, ix=1:Nx
        if x[ix] < xb1
            sx[ix,iy] = abs((x[ix] - xb1) / xpml1)^3
        end
        if x[ix] > xb2
            sx[ix,iy] = abs((x[ix] - xb2) / xpml2)^3
        end
        if y[iy] < yb1
            sy[ix,iy] = abs((y[iy] - yb1) / ypml1)^3
        end
        if y[iy] > yb2
            sy[ix,iy] = abs((y[iy] - yb2) / ypml2)^3
        end
    end

    # circle:
    # rad = xmax - xpml1
    # rmax = xmax
    # for iy=1:Ny, ix=1:Nx
    #     r = sqrt(x[ix]^2 + y[iy]^2)
    #     if r > rad
    #         sx[ix,iy] = abs((r - rad) / (rmax - rad))^3
    #         sy[ix,iy] = abs((r - rad) / (rmax - rad))^3
    #     end
    # end

    return sx, sy
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
    # xb1, xb2 = xmin + xpml1, xmax - xpml2
    # yb1, yb2 = ymin + ypml1, ymax - ypml2
    # zb1, zb2 = zmin + zpml1, zmax - zpml2
    # for iz=1:Nz, iy=1:Ny, ix=1:Nx
    #     if x[ix] < xb1
    #         sx[ix,iy,iz] = abs((x[ix] - xb1) / xpml1)^3
    #     end
    #     if x[ix] > xb2
    #         sx[ix,iy,iz] = abs((x[ix] - xb2) / xpml2)^3
    #     end
    #     if y[iy] < yb1
    #         sy[ix,iy,iz] = abs((y[iy] - yb1) / ypml1)^3
    #     end
    #     if y[iy] > yb2
    #         sy[ix,iy,iz] = abs((y[iy] - yb2) / ypml2)^3
    #     end
    #     if z[iz] < zb1
    #         sz[ix,iy,iz] = abs((z[iz] - zb1) / zpml1)^3
    #     end
    #     if z[iz] > zb2
    #         sz[ix,iy,iz] = abs((z[iz] - zb2) / zpml2)^3
    #     end
    # end

    # z cylinder:
    rad = xmax - xpml1
    rmax = xmax
    zb1, zb2 = zmin + zpml1, zmax - zpml2
    for iz=1:Nz, iy=1:Ny, ix=1:Nx
        r = sqrt(x[ix]^2 + y[iy]^2)
        if r > rad
            sx[ix,iy,iz] = abs((r - rad) / (rmax - rad))^3
            sy[ix,iy,iz] = abs((r - rad) / (rmax - rad))^3
        end
        if z[iz] < zb1
            sz[ix,iy,iz] = abs((z[iz] - zb1) / zpml1)^3
        end
        if z[iz] > zb2
            sz[ix,iy,iz] = abs((z[iz] - zb2) / zpml2)^3
        end
    end

    return sx, sy, sz
end
