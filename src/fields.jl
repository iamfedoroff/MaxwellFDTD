abstract type Field end


# ******************************************************************************
# 1D: d/dx = d/dy = 0,   (Hy, Ex)
# ******************************************************************************
struct Field1D{G, A} <: Field
    grid :: G
    # magnetic field comonents:
    Hy :: A
    # electric field displacement comonents:
    Dx :: A
    # electric field comonents:
    Ex :: A
end

@adapt_structure Field1D


function Field(grid::Grid1D)
    (; Nz) = grid
    Hy, Dx, Ex = (zeros(Nz) for i=1:3)
    return Field1D(grid, Hy, Dx, Ex)
end


function Poynting(field::Field1D)
    (; Ex, Hy) = field
    return @. sqrt((Ex*Hy)^2)
end


# ******************************************************************************
# 2D
# ******************************************************************************
struct Field2D{G, A} <: Field
    grid :: G
    # magnetic field comonents:
    Hy :: A
    # electric field displacement comonents:
    Dx :: A
    Dz :: A
    # electric field comonents:
    Ex :: A
    Ez :: A
end

@adapt_structure Field2D


function Field(grid::Grid2D)
    (; Nx, Nz) = grid
    Hy, Dx, Dz, Ex, Ez = (zeros(Nx,Nz) for i=1:5)
    return Field2D(grid, Hy, Dx, Dz, Ex, Ez)
end


function Poynting(field::Field2D)
    (; Hy, Ex, Ez) = field
    return @. sqrt((-Ez*Hy)^2 + (Ex*Hy)^2)
end


# ******************************************************************************
# 3D
# ******************************************************************************
struct Field3D{G, A} <: Field
    grid :: G
    # magnetic field comonents:
    Hx :: A
    Hy :: A
    Hz :: A
    # electric field displacement comonents:
    Dx :: A
    Dy :: A
    Dz :: A
    # electric field comonents:
    Ex :: A
    Ey :: A
    Ez :: A
end

@adapt_structure Field3D


function Field(grid::Grid3D)
    (; Nx, Ny, Nz) = grid
    Hx, Hy, Hz, Dx, Dy, Dz, Ex, Ey, Ez = (zeros(Nx,Ny,Nz) for i=1:9)
    return Field3D(grid, Hx, Hy, Hz, Dx, Dy, Dz, Ex, Ey, Ez)
end


function Poynting(field::Field3D)
    (; Hx, Hy, Hz, Ex, Ey, Ez) = field
    return @. sqrt((Ey*Hz - Ez*Hy)^2 + (Ez*Hx - Ex*Hz)^2 + (Ex*Hy - Ey*Hx)^2)
end
