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
struct Model1D{F, T, R, A, S} <: Model
    field :: F
    Nt :: Int
    dt :: T
    t :: R
    mHy1 :: A
    mHy2 :: A
    mEx1 :: A
    mEx2 :: A
    source :: S
end

@adapt_structure Model1D


function Model(field::Field1D, source; tmax, CN=1, pml_box=(0,0))
    (; grid) = field
    (; dz) = grid

    dt = CN / C0 / sqrt(1/dz^2)
    Nt = ceil(Int, tmax / dt)
    t = range(0, tmax, Nt)

    sz = pml(grid, pml_box)
    @. sz *= 1 / (2*dt)

    eps = 1
    mu = 1

    mHy0 = @. sz * dt / 2
    mHy1 = @. (1 - mHy0) / (1 + mHy0)
    mHy2 = @. -C0 * dt / mu / (1 + mHy0)
    mEx0 = @. sz * dt / 2
    mEx1 = @. (1 - mEx0) / (1 + mEx0)
    mEx2 = @. C0 * dt / eps / (1 + mEx0)

    return Model1D(field, Nt, dt, t, mHy1, mHy2, mEx1, mEx2, source)
end


function update_H!(model::Model1D)
    (; field, mHy1, mHy2) = model
    (; Hy, CEy) = field
    @. Hy = mHy1 * Hy + mHy2 * CEy
    return nothing
end


function update_E!(model::Model1D)
    (; field, mEx1, mEx2) = model
    (; Ex, CHx) = field
    @. Ex = mEx1 * Ex + mEx2 * CHx
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
    mHx1 :: A
    mHx2 :: A
    mHx3 :: A
    mHy1 :: A
    mHy2 :: A
    mHy3 :: A
    mEz1 :: A
    mEz2 :: A
    mEz3 :: A
    ICEx :: A
    ICEy :: A
    IEz :: A
    source :: S
end

@adapt_structure Model2D


function Model(field::Field2D, source; tmax, CN=1, pml_box=(0,0,0,0))
    (; grid) = field
    (; Nx, Ny, dx, dy) = grid

    dt = CN / C0 / sqrt(1/dx^2 + 1/dy^2)
    Nt = ceil(Int, tmax / dt)
    t = range(0, tmax, Nt)

    sx, sy = pml(grid, pml_box)
    @. sx *= 1 / (2*dt)
    @. sy *= 1 / (2*dt)

    eps = 1
    mu = 1

    # update coefficients:
    mHx0 = @. sy * dt / 2
    mHx1 = @. (1 - mHx0) / (1 + mHx0)
    mHx2 = @. -C0 * dt / mu / (1 + mHx0)
    mHx3 = @. -C0 * sx * dt^2 / mu / (1 + mHx0)

    mHy0 = @. sx * dt / 2
    mHy1 = @. (1 - mHy0) / (1 + mHy0)
    mHy2 = @. -C0 * dt / mu / (1 + mHy0)
    mHy3 = @. -C0 * sy * dt^2 / mu / (1 + mHy0)

    mEz0 = @. (sx + sy) * dt / 2 + sx * sy * dt^2 / 4
    mEz1 = @. (1 - mEz0) / (1 + mEz0)
    mEz2 = @. C0 * dt / eps / (1 + mEz0)
    mEz3 = @. -sx * sy * dt^2 / (1 + mEz0)

    # PML integrals:
    ICEx, ICEy, IEz = (zeros(Nx,Ny) for i=1:3)

    return Model2D(
        field, Nt, dt, t, mHx1, mHx2, mHx3, mHy1, mHy2, mHy3, mEz1, mEz2, mEz3,
        ICEx, ICEy, IEz, source,
    )
end


function update_H!(model::Model2D)
    (; field, mHx1, mHx2, mHx3, mHy1, mHy2, mHy3, ICEx, ICEy) = model
    (; Hx, Hy, CEx, CEy) = field
    @. Hx = mHx1 * Hx + mHx2 * CEx + mHx3 * ICEx
    @. Hy = mHy1 * Hy + mHy2 * CEy + mHy3 * ICEy
    return nothing
end


function update_E!(model::Model2D)
    (; field, mEz1, mEz2, mEz3, IEz) = model
    (; Ez, CHz) = field
    @. Ez = mEz1 * Ez + mEz2 * CHz + mEz3 * IEz
    return nothing
end


function update_ICE!(model::Model2D)
    (; field, ICEx, ICEy) = model
    (; CEx, CEy) = field
    @. ICEx += CEx
    @. ICEy += CEy
    return nothing
end


function update_IE!(model::Model2D)
    (; field, IEz) = model
    (; Ez) = field
    @. IEz += Ez
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
    mHx3 :: A
    mHx4 :: A
    mHy1 :: A
    mHy2 :: A
    mHy3 :: A
    mHy4 :: A
    mHz1 :: A
    mHz2 :: A
    mHz3 :: A
    mHz4 :: A
    mEx1 :: A
    mEx2 :: A
    mEx3 :: A
    mEx4 :: A
    mEy1 :: A
    mEy2 :: A
    mEy3 :: A
    mEy4 :: A
    mEz1 :: A
    mEz2 :: A
    mEz3 :: A
    mEz4 :: A
    ICHx :: A
    ICHy :: A
    ICHz :: A
    ICEx :: A
    ICEy :: A
    ICEz :: A
    IHx :: A
    IHy :: A
    IHz :: A
    IEx :: A
    IEy :: A
    IEz :: A
    source :: S
end

@adapt_structure Model3D


function Model(field::Field3D, source; tmax, CN=1, pml_box=(0,0,0,0,0,0))
    (; grid) = field
    (; Nx, Ny, Nz, dx, dy, dz) = grid

    dt = CN / C0 / sqrt(1/dx^2 + 1/dy^2 + 1/dz^2)
    Nt = ceil(Int, tmax / dt)
    t = range(0, tmax, Nt)

    sx, sy, sz = pml(grid, pml_box)
    @. sx *= 1 / (2*dt)
    @. sy *= 1 / (2*dt)
    @. sz *= 1 / (2*dt)

    eps = 1
    mu = 1

    # update coefficients:
    mHx0 = @. (sy + sz) * dt / 2 + sy * sz * dt^2 / 4
    mHx1 = @. (1 - mHx0) / (1 + mHx0)
    mHx2 = @. -C0 * dt / mu / (1 + mHx0)
    mHx3 = @. -C0 * sx * dt^2 / mu / (1 + mHx0)
    mHx4 = @. -sy * sx * dt^2 / (1 + mHx0)

    mHy0 = @. (sx + sz) * dt / 2 + sx * sz * dt^2 / 4
    mHy1 = @. (1 - mHy0) / (1 + mHy0)
    mHy2 = @. -C0 * dt / mu / (1 + mHy0)
    mHy3 = @. -C0 * sy * dt^2 / mu / (1 + mHy0)
    mHy4 = @. -sx * sz * dt^2 / (1 + mHy0)

    mHz0 = @. (sx + sy) * dt / 2 + sx * sy * dt^2 / 4
    mHz1 = @. (1 - mHz0) / (1 + mHz0)
    mHz2 = @. -C0 * dt / mu / (1 + mHz0)
    mHz3 = @. -C0 * sz * dt^2 / mu / (1 + mHz0)
    mHz4 = @. -sx * sy * dt^2 / (1 + mHz0)

    mEx0 = @. (sy + sz) * dt / 2 + sy * sz * dt^2 / 4
    mEx1 = @. (1 - mEx0) / (1 + mEx0)
    mEx2 = @. C0 * dt / eps / (1 + mEx0)
    mEx3 = @. C0 * sx * dt^2 / eps / (1 + mEx0)
    mEx4 = @. -sy * sz * dt^2 / (1 + mEx0)

    mEy0 = @. (sx + sz) * dt / 2 + sx * sz * dt^2 / 4
    mEy1 = @. (1 - mEy0) / (1 + mEy0)
    mEy2 = @. C0 * dt / eps / (1 + mEy0)
    mEy3 = @. C0 * sy * dt^2 / eps / (1 + mEy0)
    mEy4 = @. -sx * sz * dt^2 / (1 + mEy0)

    mEz0 = @. (sx + sy) * dt / 2 + sx * sy * dt^2 / 4
    mEz1 = @. (1 - mEz0) / (1 + mEz0)
    mEz2 = @. C0 * dt / eps / (1 + mEz0)
    mEz3 = @. C0 * sz * dt^2 / eps / (1 + mEz0)
    mEz4 = @. -sx * sy * dt^2 / (1 + mEz0)

    # PML integrals:
    IHx, IHy, IHz, IEx, IEy, IEz = (zeros(Nx,Ny,Nz) for i=1:6)
    ICEx, ICEy, ICEz, ICHx, ICHy, ICHz = (zeros(Nx,Ny,Nz) for i=1:6)

    return Model3D(
        field, Nt, dt, t,
        mHx1, mHx2, mHx3, mHx4, mHy1, mHy2, mHy3, mHy4, mHz1, mHz2, mHz3, mHz4,
        mEx1, mEx2, mEx3, mEx4, mEy1, mEy2, mEy3, mEy4, mEz1, mEz2, mEz3, mEz4,
        ICHx, ICHy, ICHz, ICEx, ICEy, ICEz, IHx, IHy, IHz, IEx, IEy, IEz,
        source,
    )
end


function update_H!(model::Model3D)
    (; field, ICEx, ICEy, ICEz, IHx, IHy, IHz) = model
    (; mHx1, mHx2, mHx3, mHx4,
       mHy1, mHy2, mHy3, mHy4,
       mHz1, mHz2, mHz3, mHz4) = model
    (; Hx, Hy, Hz, CEx, CEy, CEz) = field
    @. Hx = mHx1 * Hx + mHx2 * CEx + mHx3 * ICEx + mHx4 * IHx
    @. Hy = mHy1 * Hy + mHy2 * CEy + mHy3 * ICEy + mHy4 * IHy
    @. Hz = mHz1 * Hz + mHz2 * CEz + mHz3 * ICEz + mHz4 * IHz
    return nothing
end


function update_E!(model::Model3D)
    (; field, ICHx, ICHy, ICHz, IEx, IEy, IEz) = model
    (; mEx1, mEx2, mEx3, mEx4,
       mEy1, mEy2, mEy3, mEy4,
       mEz1, mEz2, mEz3, mEz4) = model
    (; Ex, Ey, Ez, CHx, CHy, CHz) = field
    @. Ex = mEx1 * Ex + mEx2 * CHx + mEx3 * ICHx + mEx4 * IEx
    @. Ey = mEy1 * Ey + mEy2 * CHy + mEy3 * ICHy + mEy4 * IEy
    @. Ez = mEz1 * Ez + mEz2 * CHz + mEz3 * ICHz + mEz4 * IEz
    return nothing
end


function update_ICH!(model::Model3D)
    (; field, ICHx, ICHy, ICHz) = model
    (; CHx, CHy, CHz) = field
    @. ICHx += CHx
    @. ICHy += CHy
    @. ICHz += CHz
    return nothing
end


function update_ICE!(model::Model3D)
    (; field, ICEx, ICEy, ICEz) = model
    (; CEx, CEy, CEz) = field
    @. ICEx += CEx
    @. ICEy += CEy
    @. ICEz += CEz
    return nothing
end


function update_IH!(model::Model3D)
    (; field, IHx, IHy, IHz) = model
    (; Hx, Hy, Hz) = field
    @. IHx += Hx
    @. IHy += Hy
    @. IHz += Hz
    return nothing
end


function update_IE!(model::Model3D)
    (; field, IEx, IEy, IEz) = model
    (; Ex, Ey, Ez) = field
    @. IEx += Ex
    @. IEy += Ey
    @. IEz += Ez
    return nothing
end
