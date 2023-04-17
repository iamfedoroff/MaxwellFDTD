abstract type Model end


function step!(model, it)
    (; field, source, t) = model

    @timeit "derivatives E" begin
        derivatives_E!(field)
        synchronize()
    end
    @timeit "update CPML E" begin
        update_CPML_E!(model)
        synchronize()
    end
    @timeit "update H" begin
        update_H!(model)
        synchronize()
    end

    @timeit "derivatives H" begin
        derivatives_H!(field)
        synchronize()
    end
    @timeit "update CPML H" begin
        update_CPML_H!(model)
        synchronize()
    end
    @timeit "update E" begin
        update_E!(model)
        synchronize()
    end

    @timeit "add_source" begin
        add_source!(field, source, t[it])  # additive source
        synchronize()
    end
    return nothing
end


function solve!(
    model; arch=CPU(), fname, nstride=nothing, nframes=nothing, dtout=nothing,
)
    model = adapt(arch, model)
    (; Nt, t) = model

    out = Output(model; fname, nstride, nframes, dtout)

    @showprogress 1 for it=1:Nt
        @timeit "model step" begin
            step!(model, it)
            synchronize()
        end

        @timeit "output" begin
            if (out.itout <= out.Ntout) && (t[it] == out.tout[out.itout])
                write_output!(out, model)
                out.itout += 1
            end
            synchronize()
        end

        if it == 1
            reset_timer!()
        end
    end

    print_timer()

    return nothing
end


# ******************************************************************************
# 1D: d/dx = d/dy = 0,   (Hy, Ex)
# ******************************************************************************
struct Model1D{F, S, T, R, A} <: Model
    field :: F
    source :: S
    Nt :: Int
    dt :: T
    t :: R
    Mh :: T
    Me :: T
    Kz :: A
    Az :: A
    Bz :: A
    psiExz :: A
    psiHyz :: A
end

@adapt_structure Model1D


function Model(
    field::Field1D, source;
    tmax,
    CN=1,
    permittivity=nothing,
    permeability=nothing,
    smooth_interfaces=true,
    pml_box=(0,0),
)
    (; grid) = field
    (; Nz, dz, z) = grid

    dt = CN / C0 / sqrt(1/dz^2)
    Nt = ceil(Int, tmax / dt)
    t = range(0, tmax, Nt)

    Kz, Az, Bz = pml(z, pml_box, dt)

    Mh = dt / MU0
    Me = dt / EPS0

    psiExz = zeros(Nz)
    psiHyz = zeros(Nz)

    return Model1D(field, source, Nt, dt, t, Mh, Me, Kz, Az, Bz, psiExz, psiHyz)
end


function update_CPML_E!(model::Model1D)
    (; field, Az, Bz, psiExz) = model
    (; dExz) = field
    @. psiExz = Bz * psiExz + Az * dExz
    return nothing
end


function update_CPML_H!(model::Model1D)
    (; field, Az, Bz, psiHyz) = model
    (; dHyz) = field
    @. psiHyz = Bz * psiHyz + Az * dHyz
    return nothing
end


function update_H!(model::Model1D)
    (; field, Mh, Kz, psiExz) = model
    (; Hy, dExz) = field
    @. Hy = Hy - Mh * (0 + dExz / Kz) - Mh * (0 + psiExz)
    return nothing
end


function update_E!(model::Model1D)
    (; field, Me, Kz, psiHyz) = model
    (; Ex, dHyz) = field
    @. Ex = Ex + Me * (0 - dHyz / Kz) + Me * (0 - psiHyz)
    return nothing
end


# ******************************************************************************
# 2D
# ******************************************************************************
struct Model2D{F, S, T, R, A1, A2} <: Model
    field :: F
    source :: S
    Nt :: Int
    dt :: T
    t :: R
    Mh :: T
    Me :: T
    Kx :: A1
    Ax :: A1
    Bx :: A1
    Kz :: A1
    Az :: A1
    Bz :: A1
    psiExz :: A2
    psiEzx :: A2
    psiHyx :: A2
    psiHyz :: A2
end

@adapt_structure Model2D


function Model(
    field::Field2D, source;
    tmax,
    CN=1,
    permittivity=nothing,
    permeability=nothing,
    smooth_interfaces=true,
    pml_box=(0,0,0,0),
)
    (; grid, w0) = field
    (; Nx, Nz, dx, dz, x, z) = grid

    dt = CN / C0 / sqrt(1/dx^2 + 1/dz^2)
    Nt = ceil(Int, tmax / dt)
    t = range(0, tmax, Nt)

    Kx, Ax, Bx = pml(x, pml_box[1:2], dt)
    Kz, Az, Bz = pml(z, pml_box[3:4], dt)

    Mh = dt / MU0
    Me = dt / EPS0

    psiExz, psiEzx, psiHyx, psiHyz = (zeros(Nx,Nz) for i=1:4)

    return Model2D(
        field, source, Nt, dt, t, Mh, Me, Kx, Ax, Bx, Kz, Az, Bz,
        psiExz, psiEzx, psiHyx, psiHyz,
    )
end


function update_CPML_E!(model::Model2D)
    (; field, Ax, Bx, Az, Bz, psiExz, psiEzx) = model
    (; dExz, dEzx) = field
    update_psi!(psiExz, Az, Bz, dExz; dim=2)
    update_psi!(psiEzx, Ax, Bx, dEzx; dim=1)
    return nothing
end


function update_CPML_H!(model::Model2D)
    (; field, Ax, Bx, Az, Bz, psiHyx, psiHyz) = model
    (; dHyx, dHyz) = field
    update_psi!(psiHyx, Ax, Bx, dHyx; dim=1)
    update_psi!(psiHyz, Az, Bz, dHyz; dim=2)
    return nothing
end


function update_H!(model::Model2D)
    (; field, Mh, Kx, Kz, psiExz, psiEzx) = model
    (; Hy, dExz, dEzx) = field
    @. Hy = Hy - Mh * (dExz - dEzx) - Mh * (psiExz - psiEzx)
    # @. Hy = Hy - Mh * (dExz / Kz - dEzx / Kx) - Mh * (psiExz - psiEzx)
    return nothing
end


function update_E!(model::Model2D)
    (; field, Me, Kx, Kz, psiHyz, psiHyx) = model
    (; Ex, Ez, dHyz, dHyx) = field
    @. Ex = Ex + Me * (0 - dHyz) + Me * (0 - psiHyz)
    @. Ez = Ez + Me * (dHyx - 0) + Me * (psiHyx - 0)
    # @. Ex = Ex + Me * (0 - dHyz / Kz) + Me * (0 - psiHyz)
    # @. Ez = Ez + Me * (dHyx / Kx - 0) + Me * (psiHyx - 0)
    return nothing
end


# ******************************************************************************
# 3D
# ******************************************************************************
struct Model3D{F, S, T, R, A1, A2} <: Model
    field :: F
    source :: S
    Nt :: Int
    dt :: T
    t :: R
    Mh :: T
    Me :: T
    Kx :: A1
    Ax :: A1
    Bx :: A1
    Ky :: A1
    Ay :: A1
    By :: A1
    Kz :: A1
    Az :: A1
    Bz :: A1
    psiExy :: A2
    psiExz :: A2
    psiEyx :: A2
    psiEyz :: A2
    psiEzx :: A2
    psiEzy :: A2
    psiHxy :: A2
    psiHxz :: A2
    psiHyx :: A2
    psiHyz :: A2
    psiHzx :: A2
    psiHzy :: A2
end

@adapt_structure Model3D


function Model(
    field::Field3D, source;
    tmax,
    CN=1,
    permittivity=nothing,
    permeability=nothing,
    smooth_interfaces=true,
    pml_box=(0,0,0,0,0,0),
)
    (; grid, w0) = field
    (; Nx, Ny, Nz, dx, dy, dz, x, y, z) = grid

    dt = CN / C0 / sqrt(1/dx^2 + 1/dy^2 + 1/dz^2)
    Nt = ceil(Int, tmax / dt)
    t = range(0, tmax, Nt)

    Kx, Ax, Bx = pml(x, pml_box[1:2], dt)
    Ky, Ay, By = pml(y, pml_box[3:4], dt)
    Kz, Az, Bz = pml(z, pml_box[5:6], dt)

    Mh = dt / MU0
    Me = dt / EPS0

    psiExy, psiExz, psiEyx, psiEyz, psiEzx, psiEzy = (zeros(Nx,Ny,Nz) for i=1:6)
    psiHxy, psiHxz, psiHyx, psiHyz, psiHzx, psiHzy = (zeros(Nx,Ny,Nz) for i=1:6)

    return Model3D(
        field, source, Nt, dt, t, Mh, Me, Kx, Ax, Bx, Ky, Ay, By, Kz, Az, Bz,
        psiExy, psiExz, psiEyx, psiEyz, psiEzx, psiEzy,
        psiHxy, psiHxz, psiHyx, psiHyz, psiHzx, psiHzy,
    )
    return nothing
end


function update_CPML_E!(model::Model3D)
    (; field, Ax, Bx, Ay, By, Az, Bz) = model
    (; psiExy, psiExz, psiEyx, psiEyz, psiEzx, psiEzy) = model
    (; dExy, dExz, dEyx, dEyz, dEzx, dEzy) = field
    update_psi!(psiExy, Ay, By, dExy; dim=2)
    update_psi!(psiExz, Az, Bz, dExz; dim=3)
    update_psi!(psiEyx, Ax, Bx, dEyx, dim=1)
    update_psi!(psiEyz, Az, Bz, dEyz; dim=3)
    update_psi!(psiEzx, Ax, Bx, dEzx; dim=1)
    update_psi!(psiEzy, Ay, By, dEzy; dim=2)
    return nothing
end


function update_CPML_H!(model::Model3D)
    (; field, Ax, Bx, Ay, By, Az, Bz) = model
    (; psiHxy, psiHxz, psiHyx, psiHyz, psiHzx, psiHzy) = model
    (; dHxy, dHxz, dHyx, dHyz, dHzx, dHzy) = field
    update_psi!(psiHxy, Ay, By, dHxy; dim=2)
    update_psi!(psiHxz, Az, Bz, dHxz; dim=3)
    update_psi!(psiHyx, Ax, Bx, dHyx, dim=1)
    update_psi!(psiHyz, Az, Bz, dHyz; dim=3)
    update_psi!(psiHzx, Ax, Bx, dHzx; dim=1)
    update_psi!(psiHzy, Ay, By, dHzy; dim=2)
    return nothing
end


function update_H!(model::Model3D)
    (; field, Mh, Kx, Ky, Kz) = model
    (; psiExy, psiExz, psiEyx, psiEyz, psiEzx, psiEzy) = model
    (; Hx, Hy, Hz, dExy, dExz, dEyx, dEyz, dEzx, dEzy) = field
    @. Hx = Hx - Mh * (dEzy - dEyz) - Mh * (psiEzy - psiEyz)
    @. Hy = Hy - Mh * (dExz - dEzx) - Mh * (psiExz - psiEzx)
    @. Hz = Hz - Mh * (dEyx - dExy) - Mh * (psiEyx - psiExy)
    # @. Hx = Hx - Mh * (dEzy / Ky - dEyz / Kz) - Mh * (psiEzy - psiEyz)
    # @. Hy = Hy - Mh * (dExz / Kz - dEzx / Kx) - Mh * (psiExz - psiEzx)
    # @. Hz = Hz - Mh * (dEyx / Kx - dExy / Ky) - Mh * (psiEyx - psiExy)
    return nothing
end


function update_E!(model::Model3D)
    (; field, Me, Kx, Ky, Kz) = model
    (; psiHxy, psiHxz, psiHyx, psiHyz, psiHzx, psiHzy) = model
    (; Ex, Ey, Ez, dHxy, dHxz, dHyx, dHyz, dHzx, dHzy) = field
    @. Ex = Ex + Me * (dHzy - dHyz) + Me * (psiHzy - psiHyz)
    @. Ey = Ey + Me * (dHxz - dHzx) + Me * (psiHxz - psiHzx)
    @. Ez = Ez + Me * (dHyx - dHxy) + Me * (psiHyx - psiHxy)
    # @. Ex = Ex + Me * (dHzy / Ky - dHyz / Kz) + Me * (psiHzy - psiHyz)
    # @. Ey = Ey + Me * (dHxz / Kz - dHzx / Kx) + Me * (psiHxz - psiHzx)
    # @. Ez = Ez + Me * (dHyx / Kx - dHxy / Ky) + Me * (psiHyx - psiHxy)
    return nothing
end


# ******************************************************************************
# Util
# ******************************************************************************
# https://julialang.org/blog/2016/02/iteration/#a_multidimensional_boxcar_filter
function moving_average(A::AbstractArray, m::Int)
    if eltype(A) == Int
        out = zeros(size(A))
    else
        out = similar(A)
    end
    R = CartesianIndices(A)
    Ifirst, Ilast = first(R), last(R)
    I1 = div(m,2) * oneunit(Ifirst)
    for I in R
        n, s = 0, zero(eltype(out))
        for J in max(Ifirst, I-I1):min(Ilast, I+I1)
            s += A[J]
            n += 1
        end
        out[I] = s/n
    end
    return out
end


function update_psi!(psiF, A, B, dF; dim)
    ci = CartesianIndices(psiF)
    for ici in eachindex(ci)
        idim = ci[ici][dim]
        psiF[ici] = B[idim] * psiF[ici] + A[idim] * dF[ici]
    end
    return nothing
end


function update_psi!(psiF::CuArray, A, B, dF; dim)
    N = length(psiF)
    @krun N update_psi_kernel!(psiF, A, B, dF, dim)
    return nothing
end
function update_psi_kernel!(psiF, A, B, dF, dim)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    ci = CartesianIndices(psiF)
    for ici=id:stride:length(ci)
        idim = ci[ici][dim]
        psiF[ici] = B[idim] * psiF[ici] + A[idim] * dF[ici]
    end
    return nothing
end
