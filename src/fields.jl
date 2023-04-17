abstract type Field end

# ******************************************************************************
# 1D: d/dx = d/dy = 0,   (Hy, Ex)
# ******************************************************************************
struct Field1D{G, T, A} <: Field
    grid :: G
    w0 :: T
    # magnetic field comonents:
    Hy :: A
    # electric field comonents:
    Ex :: A
    # electric field derivatives:
    dExz :: A
    # magnetic field derivatives:
    dHyz :: A
end

@adapt_structure Field1D


function Field(grid::Grid1D; w0)
    (; Nz) = grid
    Hy, Ex, dExz, dHyz = (zeros(Nz) for i=1:4)
    return Field1D(grid, w0, Hy, Ex, dExz, dHyz)
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
struct Field2D{G, T, A} <: Field
    grid :: G
    w0 :: T
    # magnetic field comonents:
    Hy :: A
    # electric field comonents:
    Ex :: A
    Ez :: A
    # electric field derivatives:
    dExz :: A
    dEzx :: A
    # magnetic field derivatives:
    dHyz :: A
    dHyx :: A
end

@adapt_structure Field2D


function Field(grid::Grid2D; w0)
    (; Nx, Nz) = grid
    Hy, Ex, Ez, dExz, dEzx, dHyz, dHyx = (zeros(Nx,Nz) for i=1:7)
    return Field2D(grid, w0, Hy, Ex, Ez, dExz, dEzx, dHyz, dHyx)
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


function derivatives_E!(field::Field2D{G,T,A}) where {G,T,A<:CuArray}
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
    (; grid, Hy, dHyz, dHyx) = field
    (; Nx, Nz, dx, dz) = grid

    for ix=1:Nx
        dHyz[ix,1] = (Hy[ix,1] - Hy[ix,Nz]) / dz
    end
    for iz=2:Nz
        for ix=1:Nx
            dHyz[ix,iz] = (Hy[ix,iz] - Hy[ix,iz-1]) / dz
        end
    end

    for iz=1:Nz
        dHyx[1,iz] = (Hy[1,iz] - Hy[Nx,iz]) / dx
        for ix=2:Nx
            dHyx[ix,iz] = (Hy[ix,iz] - Hy[ix-1,iz]) / dx
        end
    end

    return nothing
end


function derivatives_H!(field::Field2D{G,T,A}) where {G,T,A<:CuArray}
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
            dHyz[ix,iz] = (Hy[ix,iz] - Hy[ix,iz-1]) / dz
            dHyx[ix,iz] = (Hy[ix,iz] - Hy[ix-1,iz]) / dx
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
struct Field3D{G, T, A} <: Field
    grid :: G
    w0 :: T
    # magnetic field comonents:
    Hx :: A
    Hy :: A
    Hz :: A
    # electric field comonents:
    Ex :: A
    Ey :: A
    Ez :: A
    # electric field curl:
    CEx :: A
    CEy :: A
    CEz :: A
    # magnetic field curl:
    CHx :: A
    CHy :: A
    CHz :: A
end

@adapt_structure Field3D


function Field(grid::Grid3D; w0)
    (; Nx, Ny, Nz) = grid
    Hx, Hy, Hz, Ex, Ey, Ez = (zeros(Nx,Ny,Nz) for i=1:6)
    CEx, CEy, CEz, CHx, CHy, CHz = (zeros(Nx,Ny,Nz) for i=1:6)
    return Field3D(
        grid, w0, Hx, Hy, Hz, Ex, Ey, Ez, CEx, CEy, CEz, CHx, CHy, CHz,
    )
end


function curl_E!(field::Field3D)
    (; grid, Ex, Ey, Ez, CEx, CEy, CEz) = field
    (; Nx, Ny, Nz, dx, dy, dz) = grid

    # curl CEx:
    for iz=1:Nz-1
        for iy=1:Ny-1
            for ix=1:Nx
                CEx[ix,iy,iz] = (Ez[ix,iy+1,iz] - Ez[ix,iy,iz]) / dy -
                                (Ey[ix,iy,iz+1] - Ey[ix,iy,iz]) / dz
            end
        end
        for ix=1:Nx
            CEx[ix,Ny,iz] = (Ez[ix,1,iz] - Ez[ix,Ny,iz]) / dy -
                            (Ey[ix,Ny,iz+1] - Ey[ix,Ny,iz]) / dz
        end
    end
    for iy=1:Ny-1
        for ix=1:Nx
            CEx[ix,iy,Nz] = (Ez[ix,iy+1,Nz] - Ez[ix,iy,Nz]) / dy -
                            (Ey[ix,iy,1] - Ey[ix,iy,Nz]) / dz
        end
    end
    for ix=1:Nx
        CEx[ix,Ny,Nz] = (Ez[ix,1,Nz] - Ez[ix,Ny,Nz]) / dy -
                        (Ey[ix,Ny,1] - Ey[ix,Ny,Nz]) / dz
    end

    # curl CEy:
    for iz=1:Nz-1
        for iy=1:Ny
            for ix=1:Nx-1
                CEy[ix,iy,iz] = (Ex[ix,iy,iz+1] - Ex[ix,iy,iz]) / dz -
                                (Ez[ix+1,iy,iz] - Ez[ix,iy,iz]) / dx
            end
            CEy[Nx,iy,iz] = (Ex[Nx,iy,iz+1] - Ex[Nx,iy,iz]) / dz -
                            (Ez[1,iy,iz] - Ez[Nx,iy,iz]) / dx
        end
    end
    for iy=1:Ny
        for ix=1:Nx-1
            CEy[ix,iy,Nz] = (Ex[ix,iy,1] - Ex[ix,iy,Nz]) / dz -
                            (Ez[ix+1,iy,Nz] - Ez[ix,iy,Nz]) / dx
        end
        CEy[Nx,iy,Nz] = (Ex[Nx,iy,1] - Ex[Nx,iy,Nz]) / dz -
                        (Ez[1,iy,Nz] - Ez[Nx,iy,Nz]) / dx
    end

    # curl CEz:
    for iz=1:Nz
        for iy=1:Ny-1
            for ix=1:Nx-1
                CEz[ix,iy,iz] = (Ey[ix+1,iy,iz] - Ey[ix,iy,iz]) / dx -
                                (Ex[ix,iy+1,iz] - Ex[ix,iy,iz]) / dy
            end
            CEz[Nx,iy,iz] = (Ey[1,iy,iz] - Ey[Nx,iy,iz]) / dx -
                            (Ex[Nx,iy+1,iz] - Ex[Nx,iy,iz]) / dy
        end
        for ix=1:Nx-1
            CEz[ix,Ny,iz] = (Ey[ix+1,Ny,iz] - Ey[ix,Ny,iz]) / dx -
                            (Ex[ix,1,iz] - Ex[ix,Ny,iz]) / dy
        end
        CEz[Nx,Ny,iz] = (Ey[1,Ny,iz] - Ey[Nx,Ny,iz]) / dx -
                        (Ex[Nx,1,iz] - Ex[Nx,Ny,iz]) / dy
    end

    return nothing
end


function curl_E!(field::Field3D{G,T,A}) where {G,T,A<:CuArray}
    (; Ex) = field
    N = length(Ex)
    @krun N curl_E_kernel!(field)
    return nothing
end
function curl_E_kernel!(field::Field3D)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    (; grid, Ex, Ey, Ez, CEx, CEy, CEz) = field
    (; Nx, Ny, Nz, dx, dy, dz) = grid

    ci = CartesianIndices(Ex)
    for ici=id:stride:length(ci)
        ix = ci[ici][1]
        iy = ci[ici][2]
        iz = ci[ici][3]

        # curl CEx:
        if (iy == Ny) && (iz == Nz)
            CEx[ix,Ny,Nz] = (Ez[ix,1,Nz] - Ez[ix,Ny,Nz]) / dy -
                            (Ey[ix,Ny,1] - Ey[ix,Ny,Nz]) / dz
        elseif iy == Ny
            CEx[ix,Ny,iz] = (Ez[ix,1,iz] - Ez[ix,Ny,iz]) / dy -
                            (Ey[ix,Ny,iz+1] - Ey[ix,Ny,iz]) / dz
        elseif iz == Nz
            CEx[ix,iy,Nz] = (Ez[ix,iy+1,Nz] - Ez[ix,iy,Nz]) / dy -
                            (Ey[ix,iy,1] - Ey[ix,iy,Nz]) / dz
        else
            CEx[ix,iy,iz] = (Ez[ix,iy+1,iz] - Ez[ix,iy,iz]) / dy -
                            (Ey[ix,iy,iz+1] - Ey[ix,iy,iz]) / dz
        end

        # curl CEy:
        if (ix == Nx) && (iz == Nz)
            CEy[Nx,iy,Nz] = (Ex[Nx,iy,1] - Ex[Nx,iy,Nz]) / dz -
                            (Ez[1,iy,Nz] - Ez[Nx,iy,Nz]) / dx
        elseif ix == Nx
            CEy[Nx,iy,iz] = (Ex[Nx,iy,iz+1] - Ex[Nx,iy,iz]) / dz -
                            (Ez[1,iy,iz] - Ez[Nx,iy,iz]) / dx
        elseif iz == Nz
            CEy[ix,iy,Nz] = (Ex[ix,iy,1] - Ex[ix,iy,Nz]) / dz -
                            (Ez[ix+1,iy,Nz] - Ez[ix,iy,Nz]) / dx
        else
            CEy[ix,iy,iz] = (Ex[ix,iy,iz+1] - Ex[ix,iy,iz]) / dz -
                            (Ez[ix+1,iy,iz] - Ez[ix,iy,iz]) / dx
        end

        # curl CEz:
        if (ix == Nx) && (iy == Ny)
            CEz[Nx,Ny,iz] = (Ey[1,Ny,iz] - Ey[Nx,Ny,iz]) / dx -
                            (Ex[Nx,1,iz] - Ex[Nx,Ny,iz]) / dy
        elseif ix == Nx
            CEz[Nx,iy,iz] = (Ey[1,iy,iz] - Ey[Nx,iy,iz]) / dx -
                            (Ex[Nx,iy+1,iz] - Ex[Nx,iy,iz]) / dy
        elseif iy == Ny
            CEz[ix,Ny,iz] = (Ey[ix+1,Ny,iz] - Ey[ix,Ny,iz]) / dx -
                            (Ex[ix,1,iz] - Ex[ix,Ny,iz]) / dy
        else
            CEz[ix,iy,iz] = (Ey[ix+1,iy,iz] - Ey[ix,iy,iz]) / dx -
                            (Ex[ix,iy+1,iz] - Ex[ix,iy,iz]) / dy
        end
    end

    return nothing
end


function curl_H!(field::Field3D)
    (; grid, Hx, Hy, Hz, CHx, CHy, CHz) = field
    (; Nx, Ny, Nz, dx, dy, dz) = grid

    # curl CHx:
    for ix=1:Nx
        CHx[ix,1,1] = (Hz[ix,1,1] - Hz[ix,Ny,1]) / dy -
                      (Hy[ix,1,1] - Hy[ix,1,Nz]) / dz
    end
    for iy=2:Ny
        for ix=1:Nx
            CHx[ix,iy,1] = (Hz[ix,iy,1] - Hz[ix,iy-1,1]) / dy -
                           (Hy[ix,iy,1] - Hy[ix,iy,Nz]) / dz
        end
    end
    for iz=2:Nz
        for ix=1:Nx
            CHx[ix,1,iz] = (Hz[ix,1,iz] - Hz[ix,Ny,iz]) / dy -
                           (Hy[ix,1,iz] - Hy[ix,1,iz-1]) / dz
        end
        for iy=2:Ny
            for ix=1:Nx
                CHx[ix,iy,iz] = (Hz[ix,iy,iz] - Hz[ix,iy-1,iz]) / dy -
                                (Hy[ix,iy,iz] - Hy[ix,iy,iz-1]) / dz
            end
        end
    end

    # curl CHy:
    for iy=1:Ny
        CHy[1,iy,1] = (Hx[1,iy,1] - Hx[1,iy,Nz]) / dz -
                      (Hz[1,iy,1] - Hz[Nx,iy,1]) / dx
        for ix=2:Nx
            CHy[ix,iy,1] = (Hx[ix,iy,1] - Hx[ix,iy,Nz]) / dz -
                           (Hz[ix,iy,1] - Hz[ix-1,iy,1]) / dx
        end
    end
    for iz=2:Nz
        for iy=1:Ny
            CHy[1,iy,iz] = (Hx[1,iy,iz] - Hx[1,iy,iz-1]) / dz -
                           (Hz[1,iy,iz] - Hz[Nx,iy,iz]) / dx
            for ix=2:Nx
                CHy[ix,iy,iz] = (Hx[ix,iy,iz] - Hx[ix,iy,iz-1]) / dz -
                                (Hz[ix,iy,iz] - Hz[ix-1,iy,iz]) / dx
            end
        end
    end

    # curl CHz:
    for iz=1:Nz
        CHz[1,1,iz] = (Hy[1,1,iz] - Hy[Nx,1,iz]) / dx -
                      (Hx[1,1,iz] - Hx[1,Ny,iz]) / dy
        for ix=2:Nx
            CHz[ix,1,iz] = (Hy[ix,1,iz] - Hy[ix-1,1,iz]) / dx -
                           (Hx[ix,1,iz] - Hx[ix,Ny,iz]) / dy
        end
        for iy=2:Ny
            CHz[1,iy,iz] = (Hy[1,iy,iz] - Hy[Nx,iy,iz]) / dx -
                           (Hx[1,iy,iz] - Hx[1,iy-1,iz]) / dy
            for ix=2:Nx
                CHz[ix,iy,iz] = (Hy[ix,iy,iz] - Hy[ix-1,iy,iz]) / dx -
                                (Hx[ix,iy,iz] - Hx[ix,iy-1,iz]) / dy
            end
        end
    end

    return nothing
end


function curl_H!(field::Field3D{G,T,A}) where {G,T,A<:CuArray}
    (; Hx) = field
    N = length(Hx)
    @krun N curl_H_kernel!(field)
    return nothing
end
function curl_H_kernel!(field::Field3D)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    (; grid, Hx, Hy, Hz, CHx, CHy, CHz) = field
    (; Nx, Ny, Nz, dx, dy, dz) = grid

    ci = CartesianIndices(Hx)
    for ici=id:stride:length(ci)
        ix = ci[ici][1]
        iy = ci[ici][2]
        iz = ci[ici][3]

        # curl CHx:
        if (iy == 1) && (iz == 1)
            CHx[ix,1,1] = (Hz[ix,1,1] - Hz[ix,Ny,1]) / dy -
                          (Hy[ix,1,1] - Hy[ix,1,Nz]) / dz
        elseif iy == 1
            CHx[ix,1,iz] = (Hz[ix,1,iz] - Hz[ix,Ny,iz]) / dy -
                           (Hy[ix,1,iz] - Hy[ix,1,iz-1]) / dz
        elseif iz == 1
            CHx[ix,iy,1] = (Hz[ix,iy,1] - Hz[ix,iy-1,1]) / dy -
                           (Hy[ix,iy,1] - Hy[ix,iy,Nz]) / dz
        else
            CHx[ix,iy,iz] = (Hz[ix,iy,iz] - Hz[ix,iy-1,iz]) / dy -
                            (Hy[ix,iy,iz] - Hy[ix,iy,iz-1]) / dz
        end

        # curl CHy:
        if (ix == 1) && (iz == 1)
            CHy[1,iy,1] = (Hx[1,iy,1] - Hx[1,iy,Nz]) / dz -
                          (Hz[1,iy,1] - Hz[Nx,iy,1]) / dx
        elseif ix == 1
            CHy[1,iy,iz] = (Hx[1,iy,iz] - Hx[1,iy,iz-1]) / dz -
                           (Hz[1,iy,iz] - Hz[Nx,iy,iz]) / dx
        elseif iz == 1
            CHy[ix,iy,1] = (Hx[ix,iy,1] - Hx[ix,iy,Nz]) / dz -
                           (Hz[ix,iy,1] - Hz[ix-1,iy,1]) / dx
        else
            CHy[ix,iy,iz] = (Hx[ix,iy,iz] - Hx[ix,iy,iz-1]) / dz -
                            (Hz[ix,iy,iz] - Hz[ix-1,iy,iz]) / dx
        end

        # curl CHz:
        if (ix == 1) && (iy == 1)
            CHz[1,1,iz] = (Hy[1,1,iz] - Hy[Nx,1,iz]) / dx -
                          (Hx[1,1,iz] - Hx[1,Ny,iz]) / dy
        elseif ix == 1
            CHz[1,iy,iz] = (Hy[1,iy,iz] - Hy[Nx,iy,iz]) / dx -
                           (Hx[1,iy,iz] - Hx[1,iy-1,iz]) / dy
        elseif iy == 1
            CHz[ix,1,iz] = (Hy[ix,1,iz] - Hy[ix-1,1,iz]) / dx -
                           (Hx[ix,1,iz] - Hx[ix,Ny,iz]) / dy
        else
            CHz[ix,iy,iz] = (Hy[ix,iy,iz] - Hy[ix-1,iy,iz]) / dx -
                            (Hx[ix,iy,iz] - Hx[ix,iy-1,iz]) / dy
        end
    end

    return nothing
end


function Poynting(field::Field3D)
    (; Hx, Hy, Hz, Ex, Ey, Ez) = field
    return @. sqrt((Ey*Hz - Ez*Hy)^2 + (Ez*Hx - Ex*Hz)^2 + (Ex*Hy - Ey*Hx)^2)
end
