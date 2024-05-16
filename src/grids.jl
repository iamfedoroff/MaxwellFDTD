abstract type Grid end


"""
    Grid(;
        xmin=nothing, xmax=nothing, Nx=nothing,
        ymin=nothing, ymax=nothing, Ny=nothing,
        zmin, zmax, Nz,
    )

Create a 1-D, 2-D, or 3-D grid depending on the set of keyword arguments.

# Keywords
- `xmin::Real=nothing`: Minimum x value in meters.
- `xmax::Real=nothing`: Maximum x value in meters.
- `Nx::Int=nothing`: Number of x points.
- `ymin::Real=nothing`: Minimum y value in meters.
- `ymax::Real=nothing`: Maximum y value in meters.
- `Ny::Int=nothing`: Number of y points.
- `zmin::Real`: Minimum z value in meters.
- `zmax::Real`: Maximum z value in meters.
- `Nz::Int`: Number of z points.

---

    Grid(; zmin, zmax, Nz)

If defined only (zmin, zmax, Nz), then create a 1-D grid in z coordinate:

    +-------------+
    zmin   z   zmax

---

    Grid(; xmin, xmax, Nx, zmin, zmax, Nz)

If defined (xmin, xmax, Nx) and (zmin, zmax, Nz), then create a 2-D grid in (x,z)
coordinates:

    xmax +-------------+
         |             |
       x |             |
         |             |
    xmin +-------------+
         zmin   z   zmax

---

    Grid(; xmin, xmax, Nx, ymin, ymax, Ny, zmin, zmax, Nz)

If defined all (xmin, xmax, Nx), (ymin, ymax, Ny), and (xmin, xmax, Nx), then create a
3-D grid in (x,y,z) coordinates:

             +-------------+
            /             /|
           /             / |
          /             /  |
    zmax +-------------+   + xmax
         |             |  /
       z |             | / x
         |             |/
    zmin +-------------+ xmin
         ymax   y   ymin
"""
function Grid(;
    xmin=nothing, xmax=nothing, Nx=nothing,
    ymin=nothing, ymax=nothing, Ny=nothing,
    zmin, zmax, Nz,
)
    xargs = (xmin, xmax, Nx)
    yargs = (ymin, ymax, Ny)
    if all(isnothing.(xargs)) && all(isnothing.(yargs))
        grid = Grid1D(; zmin, zmax, Nz)
    elseif all(i->!i, isnothing.(xargs)) && all(isnothing.(yargs))
        grid = Grid2D(; xmin, xmax, Nx, zmin, zmax, Nz)
    elseif all(i->!i, isnothing.(xargs)) && all(i->!i, isnothing.(yargs))
        grid = Grid3D(; xmin, xmax, Nx, ymin, ymax, Ny, zmin, zmax, Nz)
    else
        error(
            """
            Wrong set of arguments.
                   To create a grid you need to define the following variables:
                   1-D grid:
                       zmin, zmax, Nz
                   2-D grid:
                       xmin, xmax, Nx
                       zmin, zmax, Nz
                   3-D grid:
                       xmin, xmax, Nx
                       ymin, ymax, Ny
                       zmin, zmax, Nz\
            """
        )
    end
    return grid
end


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


"""
    Grid1D(; zmin, zmax, Nz)

1-D grid in z coordinate:

    +-------------+
    zmin   z   zmax

# Keywords
- `zmin::Real`: Minimum z value in meters.
- `zmax::Real`: Maximum z value in meters.
- `Nz::Int`: Number of z points.
"""
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


"""
    Grid2D(; xmin, xmax, Nx, zmin, zmax, Nz)

2-D grid in (x,z) coordinates:

    xmax +-------------+
         |             |
       x |             |
         |             |
    xmin +-------------+
         zmin   z   zmax

# Keywords
- `xmin::Real`: Minimum x value in meters.
- `xmax::Real`: Maximum x value in meters.
- `Nx::Int`: Number of x points.
- `zmin::Real`: Minimum z value in meters.
- `zmax::Real`: Maximum z value in meters.
- `Nz::Int`: Number of z points.
"""
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


"""
    Grid3D(; xmin, xmax, Nx, ymin, ymax, Ny, zmin, zmax, Nz)

3-D grid in (x,y,z) coordinates:

             +-------------+
            /             /|
           /             / |
          /             /  |
    zmax +-------------+   + xmax
         |             |  /
       z |             | / x
         |             |/
    zmin +-------------+ xmin
         ymax   y   ymin

# Keywords
- `xmin::Real`: Minimum x value in meters.
- `xmax::Real`: Maximum x value in meters.
- `Nx::Int`: Number of x points.
- `ymin::Real`: Minimum y value in meters.
- `ymax::Real`: Maximum y value in meters.
- `Ny::Int`: Number of y points.
- `zmin::Real`: Minimum z value in meters.
- `zmax::Real`: Maximum z value in meters.
- `Nz::Int`: Number of z points.
"""
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
