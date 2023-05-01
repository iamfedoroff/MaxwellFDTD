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
struct Model1D{F, S, T, R, A, AP}
    field :: F
    source :: S
    # Time grid:
    Nt :: Int
    dt :: T
    t :: R
    # Update coefficients for H and E fields:
    Mh :: A
    Me :: A
    # Variables for conductivity calculation:
    Ms :: A
    Sx :: A
    # Variables for ADE dispersion calculation:
    Nq :: Int
    Aq :: AP
    Bq :: AP
    Cq :: AP
    Px :: AP
    oldPx1 :: AP
    oldPx2 :: AP
    # CPML variables:
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
    geometry,
    material,
    pml_box=(0,0),
)
    (; grid) = field
    (; Nz, dz, z) = grid

    # Time grid:
    dt = CN / C0 / sqrt(1/dz^2)
    Nt = ceil(Int, tmax / dt)
    t = range(0, tmax, Nt)

    # Permittivity, permeability, and conductivity:
    (; eps, mu, sigma) = material
    eps = [geometry[iz] ? eps : 1 for iz=1:Nz]
    mu = [geometry[iz] ? mu : 1 for iz=1:Nz]
    sigma = [geometry[iz] ? sigma : 0 for iz=1:Nz]

    # Update coefficients for H and E fields:
    Mh = @. dt / (MU0*mu)
    Me = @. 1 / (EPS0*eps + sigma*dt)

    # Variables for conductivity calculation:
    Ms = @. sigma * dt
    Sx = zeros(Nz)

    # Variables for ADE dispersion calculation:
    (; chi) = material
    Nq = length(chi)
    Aq, Bq, Cq = (zeros(Nq,Nz) for i=1:3)
    for iz=1:Nz, iq=1:Nq
        Aq0, Bq0, Cq0 = ade_coefficients(chi[iq], dt)
        Aq[iq,iz] = geometry[iz] * Aq0
        Bq[iq,iz] = geometry[iz] * Bq0
        Cq[iq,iz] = geometry[iz] * Cq0
    end
    Px, oldPx1, oldPx2 = (zeros(Nq,Nz) for i=1:3)


    # CPML variables:
    Kz, Az, Bz = pml(z, pml_box, dt)
    psiExz, psiHyz = zeros(Nz), zeros(Nz)

    return Model1D(
        field, source, Nt, dt, t, Mh, Me, Ms, Sx,
        Nq, Aq, Bq, Cq, Px, oldPx1, oldPx2, Kz, Az, Bz, psiExz, psiHyz,
    )
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


function update_D!(model::Model1D)
    (; field, dt, Kz, psiHyz) = model
    (; Dx, dHyz) = field
    @. Dx = Dx + dt * ((0 - dHyz/Kz) + (0 - psiHyz))
    return nothing
end


function update_P!(model::Model1D)
    (; field, Aq, Bq, Cq, Px, oldPx1, oldPx2) = model
    (; Ex) = field
    Nq, Nz = size(Px)
    for iz=1:Nz, iq=1:Nq
        oldPx2[iq,iz] = oldPx1[iq,iz]
        oldPx1[iq,iz] = Px[iq,iz]
        Px[iq,iz] = Aq[iq,iz] * Px[iq,iz] +
                    Bq[iq,iz] * oldPx2[iq,iz] +
                    Cq[iq,iz] * Ex[iz]
    end
    return nothing
end


function update_E!(model::Model1D)
    (; field, Me, Sx, Px) = model
    (; Ex, Dx) = field
    Nq, Nz = size(Px)
    for iz=1:Nz
        sumPx = zero(eltype(Px))
        for iq=1:Nq
            sumPx += Px[iq,iz]
        end
        Ex[iz] = Me[iz] * (Dx[iz] - Sx[iz] - sumPx)
    end
    return nothing
end


function update_S!(model::Model1D)
    (; field, Ms, Sx) = model
    (; Ex) = field
    @. Sx = Sx + Ms * Ex
    return nothing
end


function step!(model::Model1D, it)
    (; field, source, t) = model

    derivatives_E!(field)

    update_CPML_E!(model)

    update_H!(model)

    derivatives_H!(field)

    update_CPML_H!(model)

    update_D!(model)
    update_P!(model)
    update_E!(model)
    update_S!(model)

    add_source!(field, source, t[it])  # additive source

    return nothing
end


# ******************************************************************************
# 2D
# ******************************************************************************
abstract type Model2D end


struct Model2D_ADE_Drude{F, S, T, R, A, V} <: Model2D
    field :: F
    source :: S
    Nt :: Int
    dt :: T
    t :: R
    Mh :: A
    Me1 :: A
    Me2 :: A
    Kx :: V
    Ax :: V
    Bx :: V
    Kz :: V
    Az :: V
    Bz :: V
    psiExz :: A
    psiEzx :: A
    psiHyx :: A
    psiHyz :: A
    Aq :: A
    Bq :: A
    Jx :: A
    Jz :: A
    oldEx :: A
    oldEz :: A
end

@adapt_structure Model2D_ADE_Drude


function Model2D_ADE_Drude(
    field::Field2D, source;
    tmax,
    CN=1,
    geometry,
    material,
    pml_box=(0,0,0,0),
)
    (; grid) = field
    (; Nx, Nz, dx, dz, x, z) = grid

    dt = CN / C0 / sqrt(1/dx^2 + 1/dz^2)
    Nt = ceil(Int, tmax / dt)
    t = range(0, tmax, Nt)

    # ..........................................................................
    (; eps, mu, sigma, chi) = material
    @assert typeof(chi) <: DrudeSusceptibility
    (; wpq, gammaq) = chi

    eps = [geometry[ix,iz] ? eps : 1 for ix=1:Nx, iz=1:Nz]
    mu = [geometry[ix,iz] ? mu : 1 for ix=1:Nx, iz=1:Nz]
    sigma = [geometry[ix,iz] ? sigma : 0 for ix=1:Nx, iz=1:Nz]

    aq = gammaq
    bq = EPS0 * wpq^2
    Aq = (1 - aq * dt / 2) / (1 + aq * dt / 2)
    Bq = bq * dt / 2 / (1 + aq * dt / 2)
    Aq = @. geometry * Aq
    Bq = @. geometry * Bq
    Jx, Jz, oldEx, oldEz = (zeros(Nx,Nz) for i=1:4)
    # ..........................................................................

    Mh = @. dt / (MU0*mu)

    Me0 = @. 2*EPS0*eps + sigma*dt + dt*Bq
    Me1 = @. (2*EPS0*eps - sigma*dt - dt*Bq) / Me0
    Me2 = @. 2*dt / Me0

    Kx, Ax, Bx = pml(x, pml_box[1:2], dt)
    Kz, Az, Bz = pml(z, pml_box[3:4], dt)

    psiExz, psiEzx, psiHyx, psiHyz = (zeros(Nx,Nz) for i=1:4)

    return Model2D_ADE_Drude(
        field, source, Nt, dt, t, Mh, Me1, Me2, Kx, Ax, Bx, Kz, Az, Bz,
        psiExz, psiEzx, psiHyx, psiHyz, Aq, Bq, Jx, Jz, oldEx, oldEz,
    )
end


struct Model2D_ADE_Lorentz{F, S, T, R, A, V} <: Model2D
    field :: F
    source :: S
    Nt :: Int
    dt :: T
    t :: R
    Mh :: A
    Me1 :: A
    Me2 :: A
    Me3 :: A
    Kx :: V
    Ax :: V
    Bx :: V
    Kz :: V
    Az :: V
    Bz :: V
    psiExz :: A
    psiEzx :: A
    psiHyx :: A
    psiHyz :: A
    Aq :: A
    Bq :: A
    Cq :: A
    Jx :: A
    Jz :: A
    oldEx1 :: A
    oldEx2 :: A
    oldEz1 :: A
    oldEz2 :: A
    oldJx1 :: A
    oldJx2 :: A
    oldJz1 :: A
    oldJz2 :: A
end

@adapt_structure Model2D_ADE_Lorentz


function Model2D_ADE_Lorentz(
    field::Field2D, source;
    tmax,
    CN=1,
    geometry,
    material,
    pml_box=(0,0,0,0),
)
    (; grid) = field
    (; Nx, Nz, dx, dz, x, z) = grid

    dt = CN / C0 / sqrt(1/dx^2 + 1/dz^2)
    Nt = ceil(Int, tmax / dt)
    t = range(0, tmax, Nt)

    # ..........................................................................
    (; eps, mu, sigma, chi) = material
    @assert typeof(chi) <: LorentzSusceptibility
    (; depsq, wq, deltaq) = chi

    eps = [geometry[ix,iz] ? eps : 1 for ix=1:Nx, iz=1:Nz]
    mu = [geometry[ix,iz] ? mu : 1 for ix=1:Nx, iz=1:Nz]
    sigma = [geometry[ix,iz] ? sigma : 0 for ix=1:Nx, iz=1:Nz]

    aq = 2 * deltaq
    bq = wq^2
    cq = EPS0 * depsq * wq^2
    Aq = (2 - bq * dt^2) / (aq * dt / 2 + 1)
    Bq = (aq * dt / 2 - 1) / (aq * dt / 2 + 1)
    Cq = cq * dt / 2 / (aq * dt / 2 + 1)
    Aq = @. geometry * Aq
    Bq = @. geometry * Bq
    Cq = @. geometry * Cq
    Jx, Jz, oldEx1, oldEx2, oldEz1, oldEz2, oldJx1, oldJx2, oldJz1, oldJz2 =
        (zeros(Nx,Nz) for i=1:10)
    # ..........................................................................

    Mh = @. dt / (MU0*mu)

    Me0 = @. 2*EPS0*eps + sigma*dt + dt*Cq
    Me1 = @. (2*EPS0*eps - sigma*dt) / Me0
    Me2 = @. dt*Cq / Me0
    Me3 = @. 2*dt / Me0

    Kx, Ax, Bx = pml(x, pml_box[1:2], dt)
    Kz, Az, Bz = pml(z, pml_box[3:4], dt)

    psiExz, psiEzx, psiHyx, psiHyz = (zeros(Nx,Nz) for i=1:4)

    return Model2D_ADE_Lorentz(
        field, source, Nt, dt, t, Mh, Me1, Me2, Me3, Kx, Ax, Bx, Kz, Az, Bz,
        psiExz, psiEzx, psiHyx, psiHyz,
        Aq, Bq, Cq, Jx, Jz,
        oldEx1, oldEx2, oldEz1, oldEz2, oldJx1, oldJx2, oldJz1, oldJz2,
    )
end


struct Model2D_ADE_DrudeLorentz{F, S, T, R, A, V, AL} <: Model2D
    field :: F
    source :: S
    Nt :: Int
    dt :: T
    t :: R
    Mh :: A
    Me1 :: A
    Me2 :: A
    Me3 :: A
    Kx :: V
    Ax :: V
    Bx :: V
    Kz :: V
    Az :: V
    Bz :: V
    psiExz :: A
    psiEzx :: A
    psiHyx :: A
    psiHyz :: A
    ADq :: A
    BDq :: A
    JDx :: A
    JDz :: A
    ALq :: AL
    BLq :: AL
    CLq :: AL
    JLx :: AL
    JLz :: AL
    oldEx1 :: A
    oldEx2 :: A
    oldEz1 :: A
    oldEz2 :: A
    oldJLx1 :: AL
    oldJLx2 :: AL
    oldJLz1 :: AL
    oldJLz2 :: AL
end

@adapt_structure Model2D_ADE_DrudeLorentz


function Model2D_ADE_DrudeLorentz(
    field::Field2D, source;
    tmax,
    CN=1,
    geometry,
    material,
    pml_box=(0,0,0,0),
)
    (; grid) = field
    (; Nx, Nz, dx, dz, x, z) = grid

    dt = CN / C0 / sqrt(1/dx^2 + 1/dz^2)
    Nt = ceil(Int, tmax / dt)
    t = range(0, tmax, Nt)

    # ..........................................................................
    (; eps, mu, sigma, chi) = material
    @assert typeof(chi) <: DrudeLorentzSusceptibility
    (; wpq, gammaq, depsq, wq, deltaq) = chi

    eps = [geometry[ix,iz] ? eps : 1 for ix=1:Nx, iz=1:Nz]
    mu = [geometry[ix,iz] ? mu : 1 for ix=1:Nx, iz=1:Nz]
    sigma = [geometry[ix,iz] ? sigma : 0 for ix=1:Nx, iz=1:Nz]

    # Drude ....................................................................
    aq = gammaq
    bq = EPS0 * wpq^2
    Aq = (1 - aq * dt / 2) / (1 + aq * dt / 2)
    Bq = bq * dt / 2 / (1 + aq * dt / 2)

    ADq = @. geometry * Aq
    BDq = @. geometry * Bq

    JDx = zeros(Nx,Nz)
    JDz = zeros(Nx,Nz)

    # Lorentz ..................................................................
    aq = @. 2 * deltaq
    bq = @. wq^2
    cq = @. EPS0 * depsq * wq^2
    Aq = @. (2 - bq * dt^2) / (aq * dt / 2 + 1)
    Bq = @. (aq * dt / 2 - 1) / (aq * dt / 2 + 1)
    Cq = @. cq * dt / 2 / (aq * dt / 2 + 1)

    Nq = length(wq)
    ALq, BLq, CLq = (zeros(Nq,Nx,Nz) for i=1:3)
    for iz=1:Nz, ix=1:Nx, iq=1:Nq
        ALq[iq,ix,iz] = geometry[ix,iz] * Aq[iq]
        BLq[iq,ix,iz] = geometry[ix,iz] * Bq[iq]
        CLq[iq,ix,iz] = geometry[ix,iz] * Cq[iq]
    end
    sumCLq = dropdims(sum(CLq, dims=1); dims=1)

    oldEx1, oldEx2, oldEz1, oldEz2 = (zeros(Nx,Nz) for i=1:4)
    oldJLx1, oldJLx2, oldJLz1, oldJLz2 = (zeros(Nq,Nx,Nz) for i=1:4)
    JLx = zeros(Nq,Nx,Nz)
    JLz = zeros(Nq,Nx,Nz)
    # ..........................................................................

    Mh = @. dt / (MU0*mu)

    Me0 = @. (2*EPS0*eps + sigma*dt + dt*BDq + dt*sumCLq)
    Me1 = @. (2*EPS0*eps - sigma*dt - dt*BDq) / Me0
    Me2 = @. dt * sumCLq / Me0
    Me3 = @. 2 * dt / Me0

    Kx, Ax, Bx = pml(x, pml_box[1:2], dt)
    Kz, Az, Bz = pml(z, pml_box[3:4], dt)

    psiExz, psiEzx, psiHyx, psiHyz = (zeros(Nx,Nz) for i=1:4)

    return Model2D_ADE_DrudeLorentz(
        field, source, Nt, dt, t, Mh, Me1, Me2, Me3, Kx, Ax, Bx, Kz, Az, Bz,
        psiExz, psiEzx, psiHyx, psiHyz,
        ADq, BDq, JDx, JDz, ALq, BLq, CLq, JLx, JLz,
        oldEx1, oldEx2, oldEz1, oldEz2, oldJLx1, oldJLx2, oldJLz1, oldJLz2,
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


function update_E!(model::Model2D_ADE_Drude)
    (; field, Me1, Me2, Kx, Kz, psiHyz, psiHyx) = model
    (; Aq, Bq, Jx, Jz, oldEx, oldEz) = model
    (; Ex, Ez, dHyz, dHyx) = field
    @. oldEx = Ex
    @. oldEz = Ez
    @. Ex = Me1 * Ex + Me2 * ((0 - dHyz) + (0 - psiHyz) - (1 + Aq)/2 * Jx)
    @. Ez = Me1 * Ez + Me2 * ((dHyx - 0) + (psiHyx - 0) - (1 + Aq)/2 * Jz)
    @. Jx = Aq * Jx + Bq * (Ex - oldEx)
    @. Jz = Aq * Jz + Bq * (Ez - oldEz)
    return nothing
end


function update_E!(model::Model2D_ADE_Lorentz)
    (; field, Me1, Me2, Me3, Kx, Kz, psiHyz, psiHyx) = model
    (; Aq, Bq, Cq, Jx, Jz) = model
    (; oldEx1, oldEx2, oldEz1, oldEz2, oldJx1, oldJx2, oldJz1, oldJz2) = model
    (; Ex, Ez, dHyz, dHyx) = field
    @. oldEx2 = oldEx1
    @. oldEx1 = Ex
    @. oldEz2 = oldEz1
    @. oldEz1 = Ez
    @. oldJx2 = oldJx1
    @. oldJx1 = Jx
    @. oldJz2 = oldJz1
    @. oldJz1 = Jz
    @. Ex = Me1 * Ex +
            Me2 * oldEx2 +
            Me3 * ((0 - dHyz) + (0 - psiHyz) - ((1 + Aq)*Jx + Bq*oldJx2)/2)
    @. Ez = Me1 * Ez +
            Me2 * oldEz2 +
            Me3 * ((dHyx - 0) + (psiHyx - 0) - ((1 + Aq)*Jz + Bq*oldJz2)/2)
    @. Jx = Aq * Jx + Bq * oldJx2 + Cq * (Ex - oldEx2)
    @. Jz = Aq * Jz + Bq * oldJz2 + Cq * (Ez - oldEz2)
    return nothing
end


function update_E!(model::Model2D_ADE_DrudeLorentz)
    (; field, Me1, Me2, Me3, Kx, Kz, psiHyz, psiHyx) = model
    (; ADq, BDq, JDx, JDz) = model
    (; ALq, BLq, CLq, JLx, JLz) = model
    (; oldEx1, oldEx2, oldEz1, oldEz2, oldJLx1, oldJLx2, oldJLz1, oldJLz2) = model
    (; Ex, Ez, dHyz, dHyx) = field

    Nq, Nx, Nz = size(JLx)
    for iz=1:Nz, ix=1:Nx
        oldEx2[ix,iz] = oldEx1[ix,iz]
        oldEx1[ix,iz] = Ex[ix,iz]
        oldEz2[ix,iz] = oldEz1[ix,iz]
        oldEz1[ix,iz] = Ez[ix,iz]

        sumJLx = 0.0
        sumJLz = 0.0
        for iq=1:Nq
            oldJLx2[iq,ix,iz] = oldJLx1[iq,ix,iz]
            oldJLx1[iq,ix,iz] = JLx[iq,ix,iz]
            sumJLx += (1 + ALq[iq,ix,iz])*JLx[iq,ix,iz] + BLq[iq,ix,iz]*oldJLx2[iq,ix,iz]
            oldJLz2[iq,ix,iz] = oldJLz1[iq,ix,iz]
            oldJLz1[iq,ix,iz] = JLz[iq,ix,iz]
            sumJLz += (1 + ALq[iq,ix,iz])*JLz[iq,ix,iz] + BLq[iq,ix,iz]*oldJLz2[iq,ix,iz]
        end

        Ex[ix,iz] = Me1[ix,iz] * Ex[ix,iz] +
                    Me2[ix,iz] * oldEx2[ix,iz] +
                    Me3[ix,iz] * (
                        (0 - dHyz[ix,iz]) + (0 - psiHyz[ix,iz]) -
                        (1 + ADq[ix,iz])/2 * JDx[ix,iz] - sumJLx/2
                    )
        Ez[ix,iz] = Me1[ix,iz] * Ez[ix,iz] +
                    Me2[ix,iz] * oldEz2[ix,iz] +
                    Me3[ix,iz] * (
                        (dHyx[ix,iz] - 0) + (psiHyx[ix,iz] - 0) -
                        (1 + ADq[ix,iz])/2 * JDz[ix,iz] - sumJLz/2
                    )

        JDx[ix,iz] = ADq[ix,iz] * JDx[ix,iz] +
                     BDq[ix,iz] * (Ex[ix,iz] - oldEx1[ix,iz])
        JDz[ix,iz] = ADq[ix,iz] * JDz[ix,iz] +
                     BDq[ix,iz] * (Ez[ix,iz] - oldEz1[ix,iz])

        for iq=1:Nq
            JLx[iq,ix,iz] = ALq[iq,ix,iz] * JLx[iq,ix,iz] +
                            BLq[iq,ix,iz] * oldJLx2[iq,ix,iz] +
                            CLq[iq,ix,iz] * (Ex[ix,iz] - oldEx2[ix,iz])
            JLz[iq,ix,iz] = ALq[iq,ix,iz] * JLz[iq,ix,iz] +
                            BLq[iq,ix,iz] * oldJLz2[iq,ix,iz] +
                            CLq[iq,ix,iz] * (Ez[ix,iz] - oldEz2[ix,iz])
        end
    end

    return nothing
end


# ******************************************************************************
# 3D
# ******************************************************************************
struct Model3D{F, S, T, R, AH, AE, A1, A2} <: Model
    field :: F
    source :: S
    Nt :: Int
    dt :: T
    t :: R
    Mh :: AH
    Me1 :: AE
    Me2 :: AE
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
    conductivity=nothing,
    pml_box=(0,0,0,0,0,0),
)
    (; grid, w0) = field
    (; Nx, Ny, Nz, dx, dy, dz, x, y, z) = grid

    if isnothing(permittivity)
        eps = 1
    else
        eps = [permittivity(x[ix],y[iy],z[iz]) for ix=1:Nx, iy=1:Ny, iz=1:Nz]
    end
    if isnothing(permeability)
        mu = 1
    else
        mu = [permeability(x[ix],y[iy],z[iz]) for ix=1:Nx, iy=1:Ny, iz=1:Nz]
    end
    if isnothing(conductivity)
        sigma = 0
    else
        sigma = [conductivity(x[ix],y[iy],z[iz]) for ix=1:Nx, iy=1:Ny, iz=1:Nz]
    end

    dt = CN / C0 / sqrt(1/dx^2 + 1/dy^2 + 1/dz^2)
    Nt = ceil(Int, tmax / dt)
    t = range(0, tmax, Nt)

    Mh = @. dt / (MU0*mu)

    Me0 = @. 2*EPS0*eps + sigma*dt # + CJsum
    Me1 = @. (2*EPS0*eps - sigma*dt) / Me0
    Me2 = @. 2*dt / Me0
    # Me3 = @. CJsum / Me0

    Kx, Ax, Bx = pml(x, pml_box[1:2], dt)
    Ky, Ay, By = pml(y, pml_box[3:4], dt)
    Kz, Az, Bz = pml(z, pml_box[5:6], dt)

    psiExy, psiExz, psiEyx, psiEyz, psiEzx, psiEzy = (zeros(Nx,Ny,Nz) for i=1:6)
    psiHxy, psiHxz, psiHyx, psiHyz, psiHzx, psiHzy = (zeros(Nx,Ny,Nz) for i=1:6)

    return Model3D(
        field, source, Nt, dt, t, Mh, Me1, Me2,
        Kx, Ax, Bx, Ky, Ay, By, Kz, Az, Bz,
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
    (; field, Me1, Me2, Kx, Ky, Kz) = model
    (; psiHxy, psiHxz, psiHyx, psiHyz, psiHzx, psiHzy) = model
    (; Ex, Ey, Ez, dHxy, dHxz, dHyx, dHyz, dHzx, dHzy) = field
    @. Ex = Me1 * Ex + Me2 * (dHzy - dHyz) + Me2 * (psiHzy - psiHyz)
    @. Ey = Me1 * Ey + Me2 * (dHxz - dHzx) + Me2 * (psiHxz - psiHzx)
    @. Ez = Me1 * Ez + Me2 * (dHyx - dHxy) + Me2 * (psiHyx - psiHxy)
    # @. Ex = Me1 * Ex + Me2 * (dHzy / Ky - dHyz / Kz) + Me2 * (psiHzy - psiHyz)
    # @. Ey = Me1 * Ey + Me2 * (dHxz / Kz - dHzx / Kx) + Me2 * (psiHxz - psiHzx)
    # @. Ez = Me1 * Ez + Me2 * (dHyx / Kx - dHxy / Ky) + Me2 * (psiHyx - psiHxy)
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
