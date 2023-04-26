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
abstract type Model1D <: Model end


struct Model1D_ADE_Debye{F, S, T, R, A} <: Model1D
    field :: F
    source :: S
    Nt :: Int
    dt :: T
    t :: R
    Mh :: A
    Me1 :: A
    Me2 :: A
    Kz :: A
    Az :: A
    Bz :: A
    psiExz :: A
    psiHyz :: A
    Aq :: A
    Bq :: A
    Jx :: A
    oldEx :: A
end

@adapt_structure Model1D_ADE_Debye


function Model1D_ADE_Debye(
    field::Field1D, source;
    tmax,
    CN=1,
    geometry,
    material,
    pml_box=(0,0),
)
    (; grid) = field
    (; Nz, dz, z) = grid

    dt = CN / C0 / sqrt(1/dz^2)
    Nt = ceil(Int, tmax / dt)
    t = range(0, tmax, Nt)

    # ..........................................................................
    (; eps, mu, sigma, chi) = material
    @assert typeof(chi) <: DebyeSusceptibility
    (; depsq, tauq) = chi

    eps = [geometry[iz] ? eps : 1 for iz=1:Nz]
    mu = [geometry[iz] ? mu : 1 for iz=1:Nz]
    sigma = [geometry[iz] ? sigma : 0 for iz=1:Nz]

    aq = 1 / tauq
    bq = EPS0 * depsq / tauq
    Aq = (1 - aq * dt / 2) / (1 + aq * dt / 2)
    Bq = bq / (1 + aq * dt / 2)
    Aq = @. geometry * Aq
    Bq = @. geometry * Bq
    Jx = zeros(Nz)
    oldEx = zeros(Nz)
    # ..........................................................................

    Mh = @. dt / (MU0*mu)

    Me1 = @. (2*EPS0*eps - sigma*dt + dt*Bq) / (2*EPS0*eps + sigma*dt + dt*Bq)
    Me2 = @. 2*dt / (2*EPS0*eps + sigma*dt + dt*Bq)


    Kz, Az, Bz = pml(z, pml_box, dt)
    psiExz, psiHyz = zeros(Nz), zeros(Nz)

    return Model1D_ADE_Debye(
        field, source, Nt, dt, t, Mh, Me1, Me2, Kz, Az, Bz, psiExz, psiHyz,
        Aq, Bq, Jx, oldEx,
    )
end


struct Model1D_ADE_Drude{F, S, T, R, A} <: Model1D
    field :: F
    source :: S
    Nt :: Int
    dt :: T
    t :: R
    Mh :: A
    Me1 :: A
    Me2 :: A
    Kz :: A
    Az :: A
    Bz :: A
    psiExz :: A
    psiHyz :: A
    Aq :: A
    Bq :: A
    Jx :: A
    oldEx :: A
end

@adapt_structure Model1D_ADE_Drude


function Model1D_ADE_Drude(
    field::Field1D, source;
    tmax,
    CN=1,
    geometry,
    material,
    pml_box=(0,0),
)
    (; grid) = field
    (; Nz, dz, z) = grid

    dt = CN / C0 / sqrt(1/dz^2)
    Nt = ceil(Int, tmax / dt)
    t = range(0, tmax, Nt)

    # ..........................................................................
    (; eps, mu, sigma, chi) = material
    @assert typeof(chi) <: DrudeSusceptibility
    (; wq, gammaq) = chi

    eps = [geometry[iz] ? eps : 1 for iz=1:Nz]
    mu = [geometry[iz] ? mu : 1 for iz=1:Nz]
    sigma = [geometry[iz] ? sigma : 0 for iz=1:Nz]

    aq = gammaq
    bq = EPS0 * wq^2
    Aq = (1 - aq * dt / 2) / (1 + aq * dt / 2)
    Bq = bq * dt / 2 / (1 + aq * dt / 2)
    Aq = @. geometry * Aq
    Bq = @. geometry * Bq
    Jx = zeros(Nz)
    oldEx = zeros(Nz)
    # ..........................................................................

    Mh = @. dt / (MU0*mu)

    Me1 = @. (2*EPS0*eps - sigma*dt - dt*Bq) / (2*EPS0*eps + sigma*dt + dt*Bq)
    Me2 = @. 2*dt / (2*EPS0*eps + sigma*dt + dt*Bq)

    Kz, Az, Bz = pml(z, pml_box, dt)
    psiExz, psiHyz = zeros(Nz), zeros(Nz)

    return Model1D_ADE_Drude(
        field, source, Nt, dt, t, Mh, Me1, Me2, Kz, Az, Bz, psiExz, psiHyz,
        Aq, Bq, Jx, oldEx,
    )
end


struct Model1D_ADE_Lorentz{F, S, T, R, A} <: Model1D
    field :: F
    source :: S
    Nt :: Int
    dt :: T
    t :: R
    Mh :: A
    Me1 :: A
    Me2 :: A
    Me3 :: A
    Kz :: A
    Az :: A
    Bz :: A
    psiExz :: A
    psiHyz :: A
    Aq :: A
    Bq :: A
    Cq :: A
    Jx :: A
    oldEx1 :: A
    oldEx2 :: A
    oldJx1 :: A
    oldJx2 :: A
end

@adapt_structure Model1D_ADE_Lorentz


function Model1D_ADE_Lorentz(
    field::Field1D, source;
    tmax,
    CN=1,
    geometry,
    material,
    pml_box=(0,0),
)
    (; grid) = field
    (; Nz, dz, z) = grid

    dt = CN / C0 / sqrt(1/dz^2)
    Nt = ceil(Int, tmax / dt)
    t = range(0, tmax, Nt)

    # ..........................................................................
    (; eps, mu, sigma, chi) = material
    @assert typeof(chi) <: LorentzSusceptibility
    (; depsq, wq, deltaq) = chi

    eps = [geometry[iz] ? eps : 1 for iz=1:Nz]
    mu = [geometry[iz] ? mu : 1 for iz=1:Nz]
    sigma = [geometry[iz] ? sigma : 0 for iz=1:Nz]

    aq = 2 * deltaq
    bq = wq^2
    cq = EPS0 * depsq * wq^2
    Aq = (2 - bq * dt^2) / (aq * dt / 2 + 1)
    Bq = (aq * dt / 2 - 1) / (aq * dt / 2 + 1)
    Cq = cq * dt / 2 / (aq * dt / 2 + 1)
    Aq = @. geometry * Aq
    Bq = @. geometry * Bq
    Cq = @. geometry * Cq
    Jx = zeros(Nz)
    oldEx1, oldEx2, oldJx1, oldJx2 = (zeros(Nz) for i=1:4)
    # ..........................................................................

    Mh = @. dt / (MU0*mu)

    Me1 = @. (2*EPS0*eps - sigma*dt) / (2*EPS0*eps + sigma*dt + dt*Cq)
    Me2 = @. dt*Cq / (2*EPS0*eps + sigma*dt + dt*Cq)
    Me3 = @. 2*dt / (2*EPS0*eps + sigma*dt + dt*Cq)

    Kz, Az, Bz = pml(z, pml_box, dt)
    psiExz, psiHyz = zeros(Nz), zeros(Nz)

    return Model1D_ADE_Lorentz(
        field, source, Nt, dt, t, Mh, Me1, Me2, Me3, Kz, Az, Bz, psiExz, psiHyz,
        Aq, Bq, Cq, Jx, oldEx1, oldEx2, oldJx1, oldJx2,
    )
end


struct Model1D_ADE_LorentzMulti{F, S, T, R, A, AL} <: Model1D
    field :: F
    source :: S
    Nt :: Int
    dt :: T
    t :: R
    Mh :: A
    Me1 :: A
    Me2 :: A
    Me3 :: A
    Kz :: A
    Az :: A
    Bz :: A
    psiExz :: A
    psiHyz :: A
    ALq :: AL
    BLq :: AL
    CLq :: AL
    JLx :: AL
    oldEx1 :: A
    oldEx2 :: A
    oldJLx1 :: AL
    oldJLx2 :: AL
end

@adapt_structure Model1D_ADE_LorentzMulti


function Model1D_ADE_LorentzMulti(
    field::Field1D, source;
    tmax,
    CN=1,
    geometry,
    material,
    pml_box=(0,0),
)
    (; grid) = field
    (; Nz, dz, z) = grid

    dt = CN / C0 / sqrt(1/dz^2)
    Nt = ceil(Int, tmax / dt)
    t = range(0, tmax, Nt)

    # ..........................................................................
    (; eps, mu, sigma, chi) = material
    @assert typeof(chi) <: LorentzMultiSusceptibility
    (; depsq, wq, deltaq) = chi

    eps = [geometry[iz] ? eps : 1 for iz=1:Nz]
    mu = [geometry[iz] ? mu : 1 for iz=1:Nz]
    sigma = [geometry[iz] ? sigma : 0 for iz=1:Nz]

    aq = @. 2 * deltaq
    bq = @. wq^2
    cq = @. EPS0 * depsq * wq^2
    Aq = @. (2 - bq * dt^2) / (aq * dt / 2 + 1)
    Bq = @. (aq * dt / 2 - 1) / (aq * dt / 2 + 1)
    Cq = @. cq * dt / 2 / (aq * dt / 2 + 1)

    Nq = length(wq)
    ALq, BLq, CLq = (zeros(Nq,Nz) for i=1:3)
    for iz=1:Nz, iq=1:Nq
        ALq[iq,iz] = @. geometry[iz] * Aq[iq]
        BLq[iq,iz] = @. geometry[iz] * Bq[iq]
        CLq[iq,iz] = @. geometry[iz] * Cq[iq]
    end
    sumCLq = vec(sum(CLq; dims=1))

    oldEx1, oldEx2 = (zeros(Nz) for i=1:2)
    oldJLx1, oldJLx2 = (zeros(Nq,Nz) for i=1:2)
    JLx = zeros(Nq,Nz)
    # ..........................................................................

    Mh = @. dt / (MU0*mu)

    Me0 = @. (2 * EPS0 * eps + sigma * dt + dt * sumCLq)
    Me1 = @. (2 * EPS0 * eps - sigma * dt) / Me0
    Me2 = @. dt * sumCLq / Me0
    Me3 = @. 2 * dt / Me0

    Kz, Az, Bz = pml(z, pml_box, dt)
    psiExz, psiHyz = zeros(Nz), zeros(Nz)

    return Model1D_ADE_LorentzMulti(
        field, source, Nt, dt, t, Mh, Me1, Me2, Me3, Kz, Az, Bz, psiExz, psiHyz,
        ALq, BLq, CLq, JLx, oldEx1, oldEx2, oldJLx1, oldJLx2,
    )
end


struct Model1D_PLRC{F, S, T, R, A, AP} <: Model1D
    field :: F
    source :: S
    Nt :: Int
    dt :: T
    t :: R
    Mh :: A
    Me1 :: A
    Me2 :: A
    Kz :: A
    Az :: A
    Bz :: A
    psiExz :: A
    psiHyz :: A
    Cr :: AP
    dchi0 :: AP
    dksi0 :: AP
    PLRCx :: AP
    oldEx :: A
end

@adapt_structure Model1D_PLRC


function Model1D_PLRC(
    field::Field1D, source;
    tmax,
    CN=1,
    geometry,
    material,
    pml_box=(0,0),
)
    (; grid) = field
    (; Nz, dz, z) = grid

    dt = CN / C0 / sqrt(1/dz^2)
    Nt = ceil(Int, tmax / dt)
    t = range(0, tmax, Nt)

    # ..........................................................................
    (; eps, mu, sigma, chi) = material

    eps = [geometry[iz] ? eps : 1 for iz=1:Nz]
    mu = [geometry[iz] ? mu : 1 for iz=1:Nz]
    sigma = [geometry[iz] ? sigma : 0 for iz=1:Nz]

    if typeof(chi) <: DebyeSusceptibility
        (; depsq, tauq) = chi
        Cr = exp(-dt / tauq)
        chi0 = depsq * (1 - Cr)
        ksi0 = depsq * tauq / dt * (1 - (dt / tauq + 1) * Cr)
        dchi0 = chi0 * (1 - Cr)
        dksi0 = ksi0 * (1 - Cr)
    elseif typeof(chi) <: DrudeSusceptibility
        (; wq, gammaq) = chi
        Cr = exp(-gammaq * dt)
        chi0 = wq^2 / gammaq * dt - wq^2 / gammaq^2 * (1 - Cr)
        ksi0 = wq^2 / gammaq * dt / 2 -
            wq^2 / gammaq^3 / dt * (1 - (1 + gammaq * dt) * Cr)
        dchi0 = -wq^2 / gammaq^2 * (1 - Cr)^2
        dksi0 = -wq^2 / gammaq^3 / dt * (1 - (1 + gammaq*dt) * Cr) * (1 - Cr)
    elseif typeof(chi) <: LorentzSusceptibility
        (; depsq, wq, deltaq) = chi
        alphaq = deltaq
        betaq = sqrt(wq^2 - deltaq^2)
        gammaq = depsq * wq^2 / betaq
        arg = alphaq + 1im * betaq
        Cr = exp(-arg * dt)
        chi0 = 1im * gammaq / arg * (1 - Cr)
        ksi0 = 1im * gammaq / arg^2 * (1 - (arg * dt + 1) * Cr)
        dchi0 = chi0 * (1 - Cr)
        dksi0 = ksi0 * (1 - Cr)
    else
        error("Wrong susceptibility type.")
    end

    Cr = @. geometry * Cr
    chi0 = @. geometry * chi0
    ksi0 = @. geometry * ksi0
    dchi0 = @. geometry * dchi0
    dksi0 = @. geometry * dksi0
    PLRCx = zeros(eltype(Cr), Nz)
    oldEx = zeros(Nz)
    # ..........................................................................

    Mh = @. dt / (MU0*mu)

    Me0 = @. (eps + sigma*dt/(2*EPS0) + real(chi0) - real(ksi0))
    Me1 = @. (eps - sigma*dt/(2*EPS0) - real(ksi0)) / Me0
    Me2 = @. dt/EPS0 / Me0

    Kz, Az, Bz = pml(z, pml_box, dt)
    psiExz, psiHyz = zeros(Nz), zeros(Nz)

    return Model1D_PLRC(
        field, source, Nt, dt, t, Mh, Me1, Me2, Kz, Az, Bz, psiExz, psiHyz,
        Cr, dchi0, dksi0, PLRCx, oldEx,
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


function update_E!(model::Model1D_ADE_Debye)
    (; field, Me1, Me2, Kz, psiHyz, Aq, Bq, Jx, oldEx) = model
    (; Ex, dHyz) = field
    @. oldEx = Ex
    @. Ex = Me1 * Ex + Me2 * ((0 - dHyz/Kz) + (0 - psiHyz) - (1 + Aq)/2 * Jx)
    @. Jx = Aq * Jx + Bq * (Ex - oldEx)
    return nothing
end


function update_E!(model::Model1D_ADE_Drude)
    (; field, Me1, Me2, Kz, psiHyz, Aq, Bq, Jx, oldEx) = model
    (; Ex, dHyz) = field
    @. oldEx = Ex
    @. Ex = Me1 * Ex + Me2 * ((0 - dHyz/Kz) + (0 - psiHyz) - (1 + Aq)/2 * Jx)
    @. Jx = Aq * Jx + Bq * (Ex + oldEx)
    return nothing
end


function update_E!(model::Model1D_ADE_Lorentz)
    (; field, Me1, Me2, Me3, Kz, psiHyz) = model
    (; Aq, Bq, Cq, Jx, oldEx1, oldEx2, oldJx1, oldJx2) = model
    (; Ex, dHyz) = field
    @. oldEx2 = oldEx1
    @. oldEx1 = Ex
    @. oldJx2 = oldJx1
    @. oldJx1 = Jx
    @. Ex = Me1 * Ex +
            Me2 * oldEx2 +
            Me3 * ((0 - dHyz/Kz) + (0 - psiHyz) - ((1 + Aq)*Jx + Bq*oldJx2)/2)
    @. Jx = Aq * Jx + Bq * oldJx2 + Cq * (Ex - oldEx2)
    return nothing
end


function update_E!(model::Model1D_ADE_LorentzMulti)
    (; field, Me1, Me2, Me3, Kz, psiHyz) = model
    (; ALq, BLq, CLq, JLx, oldEx1, oldEx2, oldJLx1, oldJLx2) = model
    (; Ex, dHyz) = field
    Nq, Nz = size(JLx)
    for iz=1:Nz
        oldEx2[iz] = oldEx1[iz]
        oldEx1[iz] = Ex[iz]
        sumJLx = 0.0
        for iq=1:Nq
            oldJLx2[iq,iz] = oldJLx1[iq,iz]
            oldJLx1[iq,iz] = JLx[iq,iz]
            sumJLx += (
                (1 + ALq[iq,iz]) * JLx[iq,iz] + BLq[iq,iz] * oldJLx2[iq,iz]
            )
        end
        Ex[iz] = Me1[iz] * Ex[iz] +
                 Me2[iz] * oldEx2[iz] +
                 Me3[iz] * ((0 - dHyz[iz]/Kz[iz]) + (0 - psiHyz[iz]) - sumJLx/2)
        for iq=1:Nq
            JLx[iq,iz] = ALq[iq,iz] * JLx[iq,iz] +
                         BLq[iq,iz] * oldJLx2[iq,iz] +
                         CLq[iq,iz] * (Ex[iz] - oldEx2[iz])
        end
    end
    return nothing
end


function update_E!(model::Model1D_PLRC)
    (; field, dt, Me1, Me2, Kz, psiHyz, Cr, dchi0, dksi0, PLRCx, oldEx) = model
    (; Ex, dHyz) = field
    @. oldEx = Ex
    @. Ex = Me1 * Ex + Me2 * ((0 - dHyz/Kz) + (0 - psiHyz) + EPS0/dt * real(PLRCx))
    @. PLRCx = Cr * PLRCx + (dchi0 - dksi0) * Ex + dksi0 * oldEx
    return nothing
end


# ******************************************************************************
# 2D
# ******************************************************************************
struct Model2D{F, S, T, R, AH, AE, A1, A2} <: Model
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
    conductivity=nothing,
    pml_box=(0,0,0,0),
)
    (; grid) = field
    (; Nx, Nz, dx, dz, x, z) = grid

    if isnothing(permittivity)
        eps = 1
    else
        eps = [permittivity(x[ix],z[iz]) for ix=1:Nx, iz=1:Nz]
    end
    if isnothing(permeability)
        mu = 1
    else
        mu = [permeability(x[ix],z[iz]) for ix=1:Nx, iz=1:Nz]
    end
    if isnothing(conductivity)
        sigma = 0
    else
        sigma = [conductivity(x[ix],z[iz]) for ix=1:Nx, iz=1:Nz]
    end

    dt = CN / C0 / sqrt(1/dx^2 + 1/dz^2)
    Nt = ceil(Int, tmax / dt)
    t = range(0, tmax, Nt)

    Mh = @. dt / (MU0*mu)

    Me0 = @. 2*EPS0*eps + sigma*dt # + CJsum
    Me1 = @. (2*EPS0*eps - sigma*dt) / Me0
    Me2 = @. 2*dt / Me0
    # Me3 = @. CJsum / Me0

    Kx, Ax, Bx = pml(x, pml_box[1:2], dt)
    Kz, Az, Bz = pml(z, pml_box[3:4], dt)

    psiExz, psiEzx, psiHyx, psiHyz = (zeros(Nx,Nz) for i=1:4)

    return Model2D(
        field, source, Nt, dt, t, Mh, Me1, Me2, Kx, Ax, Bx, Kz, Az, Bz,
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
    (; field, Me1, Me2, Kx, Kz, psiHyz, psiHyx) = model
    (; Ex, Ez, dHyz, dHyx) = field
    @. Ex = Me1 * Ex + Me2 * (0 - dHyz) + Me2 * (0 - psiHyz)
    @. Ez = Me1 * Ez + Me2 * (dHyx - 0) + Me2 * (psiHyx - 0)
    # @. Ex = Me1 * Ex + Me2 * (0 - dHyz / Kz) + Me2 * (0 - psiHyz)
    # @. Ez = Me1 * Ez + Me2 * (dHyx / Kx - 0) + Me2 * (psiHyx - 0)
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
