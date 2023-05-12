abstract type Grid end


# ******************************************************************************
# 1D: d/dx = d/dy = 0
# ******************************************************************************
struct Grid1D{T, R} <: Grid
    Nz :: Int
    dz :: T
    Lz :: T
    z :: R
end

@adapt_structure Grid1D


function Grid1D(; zmin, zmax, Nz)
    z = range(zmin, zmax, Nz)
    dz = z[2] - z[1]
    Lz = z[end] - z[1]
    return Grid1D(Nz, dz, Lz, z)
end


# ******************************************************************************
# 2D: d/dy = 0
# ******************************************************************************
struct Grid2D{T, R} <: Grid
    Nx :: Int
    Nz :: Int
    dx :: T
    dz :: T
    Lx :: T
    Lz :: T
    x :: R
    z :: R
end

@adapt_structure Grid2D


function Grid2D(; xmin, xmax, Nx, zmin, zmax, Nz)
    x = range(xmin, xmax, Nx)
    z = range(zmin, zmax, Nz)
    dx = x[2] - x[1]
    dz = z[2] - z[1]
    Lx = x[end] - x[1]
    Lz = z[end] - z[1]
    return Grid2D(Nx, Nz, dx, dz, Lx, Lz, x, z)
end


# ******************************************************************************
# 3D
# ******************************************************************************
struct Grid3D{T, R} <: Grid
    Nx :: Int
    Ny :: Int
    Nz :: Int
    dx :: T
    dy :: T
    dz :: T
    Lx :: T
    Ly :: T
    Lz :: T
    x :: R
    y :: R
    z :: R
end

@adapt_structure Grid3D


function Grid3D(; xmin, xmax, Nx, ymin, ymax, Ny, zmin, zmax, Nz)
    x = range(xmin, xmax, Nx)
    y = range(ymin, ymax, Ny)
    z = range(zmin, zmax, Nz)
    dx = x[2] - x[1]
    dy = y[2] - y[1]
    dz = z[2] - z[1]
    Lx = x[end] - x[1]
    Ly = y[end] - y[1]
    Lz = z[end] - z[1]
    return Grid3D(Nx, Ny, Nz, dx, dy, dz, Lx, Ly, Lz, x, y, z)
end
