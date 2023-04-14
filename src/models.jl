abstract type Model end


function step!(model, it)
    (; field, source, t) = model

    # tfsf source:
    # (; mHy2, mEx2, Esrc, Hsrc) = model
    # (; grid, Hy, Ex) = field
    # (; dz, z) = grid
    # zsrc = 0
    # izsrc = searchsortedfirst(z, zsrc)

    @timeit "curl_E" begin
        curl_E!(field)
        synchronize()
    end
    @timeit "update_H" begin
        update_H!(model)
        synchronize()
    end
    # Esrc = source(t[it])
    # Hy[izsrc-1] -= mHy2[izsrc-1]/dz * Esrc   # tfsf source

    @timeit "curl_H" begin
        curl_H!(field)
        synchronize()
    end
    @timeit "update_E" begin
        update_E!(model)
        synchronize()
    end
    # n = sqrt(eps * mu)
    # A = sqrt(eps / mu)
    # td = n * dz / (2 * C0) + dt / 2
    # Hsrc = A * source(t[it] + td)
    # Ex[izsrc] += mEx2[izsrc]/dz * Hsrc   # tfsf source

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


function step!(model::Model1D, it)
    (; field, source, t, Mh, Me, Kz, Az, Bz, psiExz, psiHyz) = model
    (; Hy, Ex, dExz, dHyz) = field

    @timeit "derivatives E" begin
        derivative_Ex_z!(field)
        synchronize()
    end
    @timeit "update PML psiE" begin
        @. psiExz = Bz * psiExz + Az * dExz
        synchronize()
    end
    @timeit "update H" begin
        @. Hy = Hy - Mh * (0 + dExz / Kz) - Mh * (0 + psiExz)
        synchronize()
    end

    @timeit "derivatives H" begin
        derivative_Hy_z!(field)
        synchronize()
    end
    @timeit "update PML psiH" begin
        @. psiHyz = Bz * psiHyz + Az * dHyz
        synchronize()
    end
    @timeit "update E" begin
        @. Ex = Ex + Me * (0 - dHyz / Kz) + Me * (0 - psiHyz)
        synchronize()
    end

    @timeit "add_source" begin
        add_source!(field, source, t[it])  # additive source
        synchronize()
    end
    return nothing
end


# ******************************************************************************
# 2D
# ******************************************************************************
struct Model2D{F, T, R, A, S} <: Model
    field :: F
    Nt :: Int
    dt :: T
    t :: R
    mHy1 :: A
    mHy2 :: A
    mEx1 :: A
    mEx2 :: A
    mEz1 :: A
    mEz2 :: A
    source :: S
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

    sx, sz = pml(grid, pml_box)
    @. sx *= 1 / (2*dt)
    @. sz *= 1 / (2*dt)

    if isnothing(permittivity)
        eps = 1
        esigma = 0
    else
        tmp = [permittivity(x[ix], z[iz]) for ix=1:Nx, iz=1:Nz]
        if smooth_interfaces
            tmp = moving_average(tmp, 2)
        end
        eps = @. real(tmp)
        esigma = @. EPS0 * w0 * imag(tmp)
    end
    if isnothing(permeability)
        mu = 1
        msigma = 0
    else
        tmp = [permeability(x[ix], z[iz]) for ix=1:Nx, iz=1:Nz]
        if smooth_interfaces
            tmp = moving_average(tmp, 2)
        end
        mu = @. real(tmp)
        msigma = @. MU0 * w0 * imag(tmp)
    end

    # update coefficients:
    mHy0 = @. (sx + sz + msigma/(MU0*mu)) * dt/2 + sx*sz*dt^2 / 4
    mHy1 = @. (1 - mHy0) / (1 + mHy0)
    mHy2 = @. -dt/(MU0*mu) / (1 + mHy0)

    mEx0 = @. (sz + esigma/(EPS0*eps)) * dt/2 + sx*esigma*dt^2 / (4*EPS0*eps)
    mEx1 = @. (1 - mEx0) / (1 + mEx0)
    mEx2 = @. dt/(EPS0*eps) / (1 + mEx0)

    mEz0 = @. (sx + esigma/(EPS0*eps)) * dt/2 + sz*esigma*dt^2 / (4*EPS0*eps)
    mEz1 = @. (1 - mEz0) / (1 + mEz0)
    mEz2 = @. dt/(EPS0*eps) / (1 + mEz0)

    return Model2D(field, Nt, dt, t, mHy1, mHy2, mEx1, mEx2, mEz1, mEz2, source)
end


function update_H!(model::Model2D)
    (; field, mHy1, mHy2) = model
    (; Hy, CEy) = field
    @. Hy = mHy1 * Hy + mHy2 * CEy
    return nothing
end


function update_E!(model::Model2D)
    (; field, mEx1, mEx2, mEz1, mEz2) = model
    (; Ex, Ez, CHx, CHz) = field
    @. Ex = mEx1 * Ex + mEx2 * CHx
    @. Ez = mEz1 * Ez + mEz2 * CHz
    return nothing
end


# ******************************************************************************
# 3D
# ******************************************************************************
struct Model3D{F, T, R, A, S} <: Model
    field :: F
    Nt :: Int
    dt :: T
    t :: R
    mHx1 :: A
    mHx2 :: A
    mHy1 :: A
    mHy2 :: A
    mHz1 :: A
    mHz2 :: A
    mEx1 :: A
    mEx2 :: A
    mEy1 :: A
    mEy2 :: A
    mEz1 :: A
    mEz2 :: A
    source :: S
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

    sx, sy, sz = pml(grid, pml_box)
    @. sx *= 1 / (2*dt)
    @. sy *= 1 / (2*dt)
    @. sz *= 1 / (2*dt)

    if isnothing(permittivity)
        eps = 1
        esigma = 0
    else
        tmp = [permittivity(x[ix], y[iy], z[iz]) for ix=1:Nx, iy=1:Ny, iz=1:Nz]
        if smooth_interfaces
            tmp = moving_average(tmp, 2)
        end
        eps = @. real(tmp)
        esigma = @. EPS0 * w0 * imag(tmp)
    end
    if isnothing(permeability)
        mu = 1
        msigma = 0
    else
        tmp = [permeability(x[ix], y[iy], z[iz]) for ix=1:Nx, iy=1:Ny, iz=1:Nz]
        if smooth_interfaces
            tmp = moving_average(tmp, 2)
        end
        mu = @. real(tmp)
        msigma = @. MU0 * w0 * imag(tmp)
    end

    # update coefficients:
    mHx0 = @. (sy + sz + msigma / (MU0 * mu)) * dt / 2 +
              (sy * sz + sx * msigma / (MU0 * mu)) * dt^2 / 4
    mHx1 = @. (1 - mHx0) / (1 + mHx0)
    mHx2 = @. -dt / (MU0 * mu) / (1 + mHx0)

    mHy0 = @. (sx + sz + msigma / (MU0 * mu)) * dt / 2 +
              (sx * sz + sy * msigma / (MU0 * mu)) * dt^2 / 4
    mHy1 = @. (1 - mHy0) / (1 + mHy0)
    mHy2 = @. -dt / (MU0 * mu) / (1 + mHy0)

    mHz0 = @. (sx + sy + msigma / (MU0 * mu)) * dt / 2 +
              (sx * sy + sz * msigma / (MU0 * mu)) * dt^2 / 4
    mHz1 = @. (1 - mHz0) / (1 + mHz0)
    mHz2 = @. -dt / (MU0 * mu) / (1 + mHz0)

    mEx0 = @. (sy + sz + esigma / (EPS0 * eps)) * dt / 2 +
              (sy * sz + sx * esigma / (EPS0 * eps)) * dt^2 / 4
    mEx1 = @. (1 - mEx0) / (1 + mEx0)
    mEx2 = @. dt / (EPS0 * eps) / (1 + mEx0)

    mEy0 = @. (sx + sz + esigma / (EPS0 * eps)) * dt / 2 +
              (sx * sz + sy * esigma / (EPS0 * eps)) * dt^2 / 4
    mEy1 = @. (1 - mEy0) / (1 + mEy0)
    mEy2 = @. dt / (EPS0 * eps) / (1 + mEy0)

    mEz0 = @. (sx + sy + esigma / (EPS0 * eps)) * dt / 2 +
              (sx * sy + sz * esigma / (EPS0 * eps)) * dt^2 / 4
    mEz1 = @. (1 - mEz0) / (1 + mEz0)
    mEz2 = @. dt / (EPS0 * eps) / (1 + mEz0)

    return Model3D(
        field, Nt, dt, t,
        mHx1, mHx2, mHy1, mHy2, mHz1, mHz2, mEx1, mEx2, mEy1, mEy2, mEz1, mEz2,
        source,
    )
end


function update_H!(model::Model3D)
    (; field) = model
    (; mHx1, mHx2, mHy1, mHy2, mHz1, mHz2) = model
    (; Hx, Hy, Hz, CEx, CEy, CEz) = field
    @. Hx = mHx1 * Hx + mHx2 * CEx
    @. Hy = mHy1 * Hy + mHy2 * CEy
    @. Hz = mHz1 * Hz + mHz2 * CEz
    return nothing
end


function update_E!(model::Model3D)
    (; field) = model
    (; mEx1, mEx2, mEy1, mEy2, mEz1, mEz2) = model
    (; Ex, Ey, Ez, CHx, CHy, CHz) = field
    @. Ex = mEx1 * Ex + mEx2 * CHx
    @. Ey = mEy1 * Ey + mEy2 * CHy
    @. Ez = mEz1 * Ez + mEz2 * CHz
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
