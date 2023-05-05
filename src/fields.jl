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


function derivatives_E!(field::Field1D)
    (; grid, Ex, dExz) = field
    (; Nz, dz) = grid
    for iz=1:Nz-1
        dExz[iz] = (Ex[iz+1] - Ex[iz]) / dz
    end
    dExz[Nz] = (Ex[1] - Ex[Nz]) / dz   # periodic bc
    return nothing
end


function derivatives_H!(field::Field1D)
    (; grid, Hy, dHyz) = field
    (; Nz, dz) = grid
    dHyz[1] = (Hy[1] - Hy[Nz]) / dz   # periodic bc
    for iz=2:Nz
        dHyz[iz] = (Hy[iz] - Hy[iz-1]) / dz
    end
    return nothing
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


function derivatives_E!(field::Field2D)
    (; grid, Ex, Ez, dExz, dEzx) = field
    (; Nx, Nz, dx, dz) = grid

    for iz=1:Nz-1
        for ix=1:Nx
            dExz[ix,iz] = (Ex[ix,iz+1] - Ex[ix,iz]) / dz
        end
    end
    for ix=1:Nx
        dExz[ix,Nz] = (Ex[ix,1] - Ex[ix,Nz]) / dz
    end

    for iz=1:Nz
        for ix=1:Nx-1
            dEzx[ix,iz] = (Ez[ix+1,iz] - Ez[ix,iz]) / dx
        end
        dEzx[Nx,iz] = (Ez[1,iz] - Ez[Nx,iz]) / dx
    end

    return nothing
end


function derivatives_E!(field::Field2D{G,A}) where {G,A<:CuArray}
    (; Ex) = field
    N = length(Ex)
    @krun N derivatives_E_kernel!(field)
    return nothing
end
function derivatives_E_kernel!(field::Field2D)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    (; grid, Ex, Ez, dExz, dEzx) = field
    (; Nx, Nz, dx, dz) = grid

    ci = CartesianIndices(Ex)
    for ici=id:stride:length(ci)
        ix = ci[ici][1]
        iz = ci[ici][2]
        if ix == Nx
            dEzx[Nx,iz] = (Ez[1,iz] - Ez[Nx,iz]) / dx
        elseif iz == Nz
            dExz[ix,Nz] = (Ex[ix,1] - Ex[ix,Nz]) / dz
        else
            dExz[ix,iz] = (Ex[ix,iz+1] - Ex[ix,iz]) / dz
            dEzx[ix,iz] = (Ez[ix+1,iz] - Ez[ix,iz]) / dx
        end
    end
    return nothing
end


function derivatives_H!(field::Field2D)
    (; grid, Hy, dHyx, dHyz) = field
    (; Nx, Nz, dx, dz) = grid

    for iz=1:Nz
        dHyx[1,iz] = (Hy[1,iz] - Hy[Nx,iz]) / dx
        for ix=2:Nx
            dHyx[ix,iz] = (Hy[ix,iz] - Hy[ix-1,iz]) / dx
        end
    end

    for ix=1:Nx
        dHyz[ix,1] = (Hy[ix,1] - Hy[ix,Nz]) / dz
    end
    for iz=2:Nz
        for ix=1:Nx
            dHyz[ix,iz] = (Hy[ix,iz] - Hy[ix,iz-1]) / dz
        end
    end

    return nothing
end


function derivatives_H!(field::Field2D{G,A}) where {G,A<:CuArray}
    (; Hy) = field
    N = length(Hy)
    @krun N derivatives_H_kernel!(field)
    return nothing
end
function derivatives_H_kernel!(field::Field2D)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    (; grid, Hy, dHyz, dHyx) = field
    (; Nx, Nz, dx, dz) = grid

    ci = CartesianIndices(Hy)
    for ici=id:stride:length(ci)
        ix = ci[ici][1]
        iz = ci[ici][2]
        if ix == 1
            dHyx[1,iz] = (Hy[1,iz] - Hy[Nx,iz]) / dx
        elseif iz == 1
            dHyz[ix,1] = (Hy[ix,1] - Hy[ix,Nz]) / dz
        else
            dHyx[ix,iz] = (Hy[ix,iz] - Hy[ix-1,iz]) / dx
            dHyz[ix,iz] = (Hy[ix,iz] - Hy[ix,iz-1]) / dz
        end
    end
    return nothing
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


function derivatives_E!(field::Field3D)
    (; grid, Ex, Ey, Ez, dExy, dExz, dEyx, dEyz, dEzx, dEzy) = field
    (; Nx, Ny, Nz, dx, dy, dz) = grid
    for iz=1:Nz, iy=1:Ny, ix=1:Nx
        if ix == Nx
            dEyx[Nx,iy,iz] = (Ey[1,iy,iz] - Ey[Nx,iy,iz]) / dx
            dEzx[Nx,iy,iz] = (Ez[1,iy,iz] - Ez[Nx,iy,iz]) / dx
        elseif iy == Ny
            dExy[ix,Ny,iz] = (Ex[ix,1,iz] - Ex[ix,Ny,iz]) / dy
            dEzy[ix,Ny,iz] = (Ez[ix,1,iz] - Ez[ix,Ny,iz]) / dy
        elseif iz == Nz
            dExz[ix,iy,Nz] = (Ex[ix,iy,1] - Ex[ix,iy,Nz]) / dz
            dEyz[ix,iy,Nz] = (Ey[ix,iy,1] - Ey[ix,iy,Nz]) / dz
        else
            dExy[ix,iy,iz] = (Ex[ix,iy+1,iz] - Ex[ix,iy,iz]) / dy
            dExz[ix,iy,iz] = (Ex[ix,iy,iz+1] - Ex[ix,iy,iz]) / dz
            dEyx[ix,iy,iz] = (Ey[ix+1,iy,iz] - Ey[ix,iy,iz]) / dx
            dEyz[ix,iy,iz] = (Ey[ix,iy,iz+1] - Ey[ix,iy,iz]) / dz
            dEzx[ix,iy,iz] = (Ez[ix+1,iy,iz] - Ez[ix,iy,iz]) / dx
            dEzy[ix,iy,iz] = (Ez[ix,iy+1,iz] - Ez[ix,iy,iz]) / dy
        end
    end
    return nothing
end


function derivatives_E!(field::Field3D{G,A}) where {G,A<:CuArray}
    (; Ex) = field
    N = length(Ex)
    @krun N derivatives_E_kenel!(field)
    return nothing
end
function derivatives_E_kenel!(field::Field3D)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    (; grid, Ex, Ey, Ez, dExy, dExz, dEyx, dEyz, dEzx, dEzy) = field
    (; Nx, Ny, Nz, dx, dy, dz) = grid

    ci = CartesianIndices(Ex)
    for ici=id:stride:length(ci)
        ix = ci[ici][1]
        iy = ci[ici][2]
        iz = ci[ici][3]
        if ix == Nx
            dEyx[Nx,iy,iz] = (Ey[1,iy,iz] - Ey[Nx,iy,iz]) / dx
            dEzx[Nx,iy,iz] = (Ez[1,iy,iz] - Ez[Nx,iy,iz]) / dx
        elseif iy == Ny
            dExy[ix,Ny,iz] = (Ex[ix,1,iz] - Ex[ix,Ny,iz]) / dy
            dEzy[ix,Ny,iz] = (Ez[ix,1,iz] - Ez[ix,Ny,iz]) / dy
        elseif iz == Nz
            dExz[ix,iy,Nz] = (Ex[ix,iy,1] - Ex[ix,iy,Nz]) / dz
            dEyz[ix,iy,Nz] = (Ey[ix,iy,1] - Ey[ix,iy,Nz]) / dz
        else
            dExy[ix,iy,iz] = (Ex[ix,iy+1,iz] - Ex[ix,iy,iz]) / dy
            dExz[ix,iy,iz] = (Ex[ix,iy,iz+1] - Ex[ix,iy,iz]) / dz
            dEyx[ix,iy,iz] = (Ey[ix+1,iy,iz] - Ey[ix,iy,iz]) / dx
            dEyz[ix,iy,iz] = (Ey[ix,iy,iz+1] - Ey[ix,iy,iz]) / dz
            dEzx[ix,iy,iz] = (Ez[ix+1,iy,iz] - Ez[ix,iy,iz]) / dx
            dEzy[ix,iy,iz] = (Ez[ix,iy+1,iz] - Ez[ix,iy,iz]) / dy
        end
    end
    return nothing
end


function derivatives_H!(field::Field3D)
    (; grid, Hx, Hy, Hz, dHxy, dHxz, dHyx, dHyz, dHzx, dHzy) = field
    (; Nx, Ny, Nz, dx, dy, dz) = grid
    for iz=1:Nz, iy=1:Ny, ix=1:Nx
        if ix == 1
            dHyx[1,iy,iz] = (Hy[1,iy,iz] - Hy[Nx,iy,iz]) / dx
            dHzx[1,iy,iz] = (Hz[1,iy,iz] - Hz[Nx,iy,iz]) / dx
        elseif iy == 1
            dHxy[ix,1,iz] = (Hx[ix,1,iz] - Hx[ix,Ny,iz]) / dy
            dHzy[ix,1,iz] = (Hz[ix,1,iz] - Hz[ix,Ny,iz]) / dy
        elseif iz == 1
            dHxz[ix,iy,1] = (Hx[ix,iy,1] - Hx[ix,iy,Nz]) / dz
            dHyz[ix,iy,1] = (Hy[ix,iy,1] - Hy[ix,iy,Nz]) / dz
        else
            dHxy[ix,iy,iz] = (Hx[ix,iy,iz] - Hx[ix,iy-1,iz]) / dy
            dHxz[ix,iy,iz] = (Hx[ix,iy,iz] - Hx[ix,iy,iz-1]) / dz
            dHyx[ix,iy,iz] = (Hy[ix,iy,iz] - Hy[ix-1,iy,iz]) / dx
            dHyz[ix,iy,iz] = (Hy[ix,iy,iz] - Hy[ix,iy,iz-1]) / dz
            dHzx[ix,iy,iz] = (Hz[ix,iy,iz] - Hz[ix-1,iy,iz]) / dx
            dHzy[ix,iy,iz] = (Hz[ix,iy,iz] - Hz[ix,iy-1,iz]) / dy
        end
    end
    return nothing
end


function derivatives_H!(field::Field3D{G,A}) where {G,A<:CuArray}
    (; Hx) = field
    N = length(Hx)
    @krun N derivatives_H_kernel!(field)
    return nothing
end
function derivatives_H_kernel!(field::Field3D)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    (; grid, Hx, Hy, Hz, dHxy, dHxz, dHyx, dHyz, dHzx, dHzy) = field
    (; Nx, Ny, Nz, dx, dy, dz) = grid

    ci = CartesianIndices(Hx)
    for ici=id:stride:length(ci)
        ix = ci[ici][1]
        iy = ci[ici][2]
        iz = ci[ici][3]
        if ix == 1
            dHyx[1,iy,iz] = (Hy[1,iy,iz] - Hy[Nx,iy,iz]) / dx
            dHzx[1,iy,iz] = (Hz[1,iy,iz] - Hz[Nx,iy,iz]) / dx
        elseif iy == 1
            dHxy[ix,1,iz] = (Hx[ix,1,iz] - Hx[ix,Ny,iz]) / dy
            dHzy[ix,1,iz] = (Hz[ix,1,iz] - Hz[ix,Ny,iz]) / dy
        elseif iz == 1
            dHxz[ix,iy,1] = (Hx[ix,iy,1] - Hx[ix,iy,Nz]) / dz
            dHyz[ix,iy,1] = (Hy[ix,iy,1] - Hy[ix,iy,Nz]) / dz
        else
            dHxy[ix,iy,iz] = (Hx[ix,iy,iz] - Hx[ix,iy-1,iz]) / dy
            dHxz[ix,iy,iz] = (Hx[ix,iy,iz] - Hx[ix,iy,iz-1]) / dz
            dHyx[ix,iy,iz] = (Hy[ix,iy,iz] - Hy[ix-1,iy,iz]) / dx
            dHyz[ix,iy,iz] = (Hy[ix,iy,iz] - Hy[ix,iy,iz-1]) / dz
            dHzx[ix,iy,iz] = (Hz[ix,iy,iz] - Hz[ix-1,iy,iz]) / dx
            dHzy[ix,iy,iz] = (Hz[ix,iy,iz] - Hz[ix,iy-1,iz]) / dy
        end
    end
    return nothing
end


function Poynting(field::Field3D)
    (; Hx, Hy, Hz, Ex, Ey, Ez) = field
    return @. sqrt((Ey*Hz - Ez*Hy)^2 + (Ez*Hx - Ex*Hz)^2 + (Ex*Hy - Ey*Hx)^2)
end
