function geometry2bool(geometry, grid::Grid1D)
    return [Bool(geometry[iz]) for iz=1:grid.Nz]
end


function geometry2bool(geometry::Function, grid::Grid1D)
    return [Bool(geometry(zi)) for zi=grid.z]
end


# ------------------------------------------------------------------------------------------
function geometry2bool(geometry, grid::Grid2D)
    return [Bool(geometry[ix,iz]) for ix=1:grid.Nx, iz=1:grid.Nz]
end


function geometry2bool(geometry::Function, grid::Grid2D)
    return [Bool(geometry(xi,zi)) for xi=grid.x, zi=grid.z]

end


# ------------------------------------------------------------------------------------------
function geometry2bool(geometry, grid::Grid3D)
    return [Bool(geometry[ix,iy,iz]) for ix=1:grid.Nx, iy=1:grid.Ny, iz=1:grid.Nz]
end


function geometry2bool(geometry::Function, grid::Grid3D)
    return [Bool(geometry(xi,yi,zi)) for xi=grid.x, yi=grid.y, zi=grid.z]
end


# ******************************************************************************************
function geometry2indices(geometry, grid::Grid1D)
    (; Nz) = grid
    return [CartesianIndex(iz) for iz=1:Nz if Bool(geometry[iz])]
end


function geometry2indices(geometry::Function, grid::Grid1D)
    (; Nz, z) = grid
    [CartesianIndex(iz) for iz=1:Nz if geometry(z[iz])]
end


# ------------------------------------------------------------------------------------------
function geometry2indices(geometry, grid::Grid2D)
    (; Nx, Nz) = grid
    return [CartesianIndex(ix,iz) for ix=1:Nx, iz=1:Nz if Bool(geometry[ix,iz])]
end


function geometry2indices(geometry::Function, grid::Grid2D)
    (; Nx, Nz, x, z) = grid
    return [CartesianIndex(ix,iz) for ix=1:Nx, iz=1:Nz if geometry(x[ix],z[iz])]
end


# ------------------------------------------------------------------------------------------
function geometry2indices(geometry, grid::Grid3D)
    (; Nx, Ny, Nz) = grid
    return [
        CartesianIndex(ix,iy,iz) for ix=1:Nx, iy=1:Ny, iz=1:Nz
        if Bool(geometry[ix,iy,iz])
    ]
end


function geometry2indices(geometry::Function, grid::Grid3D)
    (; Nx, Ny, Nz, x, y, z) = grid
    return [
        CartesianIndex(ix,iy,iz) for ix=1:Nx, iy=1:Ny, iz=1:Nz
        if geometry(x[ix],y[iy],z[iz])
    ]
end
