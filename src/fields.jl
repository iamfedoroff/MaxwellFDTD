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
    # electric field derivatives:
    dExz :: A
    # magnetic field derivatives:
    dHyz :: A
end

@adapt_structure Field1D


function Field(grid::Grid1D)
    (; Nz) = grid
    Hy, Dx, Ex, dExz, dHyz = (zeros(Nz) for i=1:5)
    return Field1D(grid, Hy, Dx, Ex, dExz, dHyz)
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
    # electric field derivatives:
    dExz :: A
    dEzx :: A
    # magnetic field derivatives:
    dHyx :: A
    dHyz :: A
end

@adapt_structure Field2D


function Field(grid::Grid2D)
    (; Nx, Nz) = grid
    Hy, Dx, Dz, Ex, Ez, dExz, dEzx, dHyx, dHyz = (zeros(Nx,Nz) for i=1:9)
    return Field2D(grid, Hy, Dx, Dz, Ex, Ez, dExz, dEzx, dHyx, dHyz)
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
    # electric field derivatives:
    dExy :: A
    dExz :: A
    dEyx :: A
    dEyz :: A
    dEzx :: A
    dEzy :: A
    # magnetic field derivatives:
    dHxy :: A
    dHxz :: A
    dHyx :: A
    dHyz :: A
    dHzx :: A
    dHzy :: A
end

@adapt_structure Field3D


function Field(grid::Grid3D)
    (; Nx, Ny, Nz) = grid
    Hx, Hy, Hz, Dx, Dy, Dz, Ex, Ey, Ez = (zeros(Nx,Ny,Nz) for i=1:9)
    dExy, dExz, dEyx, dEyz, dEzx, dEzy = (zeros(Nx,Ny,Nz) for i=1:6)
    dHxy, dHxz, dHyx, dHyz, dHzx, dHzy = (zeros(Nx,Ny,Nz) for i=1:6)
    return Field3D(
        grid, Hx, Hy, Hz, Dx, Dy, Dz, Ex, Ey, Ez,
        dExy, dExz, dEyx, dEyz, dEzx, dEzy,
        dHxy, dHxz, dHyx, dHyz, dHzx, dHzy,
    )
end


function Poynting(field::Field3D)
    (; Hx, Hy, Hz, Ex, Ey, Ez) = field
    return @. sqrt((Ey*Hz - Ez*Hy)^2 + (Ez*Hx - Ex*Hz)^2 + (Ex*Hy - Ey*Hx)^2)
end
