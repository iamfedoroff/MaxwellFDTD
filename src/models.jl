abstract type Model end


# function step!(model, it)
#     (; field, source, t) = model

#     @timeit "derivatives E" begin
#         derivatives_E!(field)
#         synchronize()
#     end
#     @timeit "update CPML E" begin
#         update_CPML_E!(model)
#         synchronize()
#     end
#     @timeit "update H" begin
#         update_H!(model)
#         synchronize()
#     end

#     @timeit "derivatives H" begin
#         derivatives_H!(field)
#         synchronize()
#     end
#     @timeit "update CPML H" begin
#         update_CPML_H!(model)
#         synchronize()
#     end
#     @timeit "update D" begin
#         update_D!(model)
#         synchronize()
#     end
#     @timeit "update P" begin
#         update_P!(model)
#         synchronize()
#     end
#     @timeit "update E" begin
#         update_E!(model)
#         synchronize()
#     end
#     @timeit "update S" begin
#         update_S!(model)
#         synchronize()
#     end

#     @timeit "add_source" begin
#         add_source!(field, source, t[it])  # additive source
#         synchronize()
#     end
#     return nothing
# end


function step!(model, it)
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

    # update_E_explicit!(model)

    add_source!(field, source, t[it])

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
    eps = [geometry(z[iz]) ? eps : 1 for iz=1:Nz]
    mu = [geometry(z[iz]) ? mu : 1 for iz=1:Nz]
    sigma = [geometry(z[iz]) ? sigma : 0 for iz=1:Nz]

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
        Aq[iq,iz] = geometry(z[iz]) * Aq0
        Bq[iq,iz] = geometry(z[iz]) * Bq0
        Cq[iq,iz] = geometry(z[iz]) * Cq0
    end
    Px, oldPx1, oldPx2 = (zeros(Nq,Nz) for i=1:3)

    # CPML variables:
    Kz, Az, Bz = pml(z, pml_box, dt)
    psiExz, psiHyz = zeros(Nz), zeros(Nz)

    return Model1D(
        field, source, Nt, dt, t, Mh, Me, Ms, Sx,
        Aq, Bq, Cq, Px, oldPx1, oldPx2, Kz, Az, Bz, psiExz, psiHyz,
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


function update_E_explicit!(model::Model1D)
    (; field, dt, Me, Kz, psiHyz) = model
    (; Ex, dHyz) = field
    @. Ex = Ex + dt * Me * ((0 - dHyz/Kz) + (0 - psiHyz))
    return nothing
end


function update_S!(model::Model1D)
    (; field, Ms, Sx) = model
    (; Ex) = field
    @. Sx = Sx + Ms * Ex
    return nothing
end


# ******************************************************************************
# 2D
# ******************************************************************************
struct Model2D{F, S, T, R, A, AP, V}
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
    Sz :: A
    # Variables for ADE dispersion calculation:
    Aq :: AP
    Bq :: AP
    Cq :: AP
    Px :: AP
    oldPx1 :: AP
    oldPx2 :: AP
    Pz :: AP
    oldPz1 :: AP
    oldPz2 :: AP
    # CPML variables:
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
end

@adapt_structure Model2D


function Model(
    field::Field2D, source;
    tmax,
    CN=1,
    geometry,
    material,
    pml_box=(0,0,0,0),
)
    (; grid) = field
    (; Nx, Nz, dx, dz, x, z) = grid

    # Time grid:
    dt = CN / C0 / sqrt(1/dx^2 + 1/dz^2)
    Nt = ceil(Int, tmax / dt)
    t = range(0, tmax, Nt)

    # Permittivity, permeability, and conductivity:
    (; eps, mu, sigma) = material
    eps = [geometry(x[ix],z[iz]) ? eps : 1 for ix=1:Nx, iz=1:Nz]
    mu = [geometry(x[ix],z[iz]) ? mu : 1 for ix=1:Nx, iz=1:Nz]
    sigma = [geometry(x[ix],z[iz]) ? sigma : 0 for ix=1:Nx, iz=1:Nz]

    # Update coefficients for H and E fields:
    Mh = @. dt / (MU0*mu)
    Me = @. 1 / (EPS0*eps + sigma*dt)

    # Variables for conductivity calculation:
    Ms = @. sigma * dt
    Sx = zeros(Nx,Nz)
    Sz = zeros(Nx,Nz)

    # Variables for ADE dispersion calculation:
    (; chi) = material
    Nq = length(chi)
    Aq, Bq, Cq = (zeros(Nq,Nx,Nz) for i=1:3)
    for iz=1:Nz, ix=1:Nx, iq=1:Nq
        Aq0, Bq0, Cq0 = ade_coefficients(chi[iq], dt)
        Aq[iq,ix,iz] = geometry(x[ix],z[iz]) * Aq0
        Bq[iq,ix,iz] = geometry(x[ix],z[iz]) * Bq0
        Cq[iq,ix,iz] = geometry(x[ix],z[iz]) * Cq0
    end
    Px, oldPx1, oldPx2 = (zeros(Nq,Nx,Nz) for i=1:3)
    Pz, oldPz1, oldPz2 = (zeros(Nq,Nx,Nz) for i=1:3)

    # CPML variables:
    Kx, Ax, Bx = pml(x, pml_box[1:2], dt)
    Kz, Az, Bz = pml(z, pml_box[3:4], dt)
    psiExz, psiEzx, psiHyx, psiHyz = (zeros(Nx,Nz) for i=1:4)

    return Model2D(
        field, source, Nt, dt, t, Mh, Me, Ms, Sx, Sz,
        Aq, Bq, Cq, Px, oldPx1, oldPx2, Pz, oldPz1, oldPz2,
        Kx, Ax, Bx, Kz, Az, Bz, psiExz, psiEzx, psiHyx, psiHyz,
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


function update_D!(model::Model2D)
    (; field, dt, Kx, Kz, psiHyx, psiHyz) = model
    (; Dx, Dz, dHyx, dHyz) = field
    @. Dx = Dx + dt * ((0 - dHyz) + (0 - psiHyz))
    @. Dz = Dz + dt * ((dHyx - 0) + (psiHyx - 0))
    return nothing
end


function update_P!(model::Model2D)
    (; field, Aq, Bq, Cq, Px, oldPx1, oldPx2, Pz, oldPz1, oldPz2) = model
    (; Ex, Ez) = field
    Nq, Nx, Nz = size(Px)
    for iz=1:Nz, ix=1:Nx, iq=1:Nq
        oldPx2[iq,ix,iz] = oldPx1[iq,ix,iz]
        oldPx1[iq,ix,iz] = Px[iq,ix,iz]
        Px[iq,ix,iz] = Aq[iq,ix,iz] * Px[iq,ix,iz] +
                       Bq[iq,ix,iz] * oldPx2[iq,ix,iz] +
                       Cq[iq,ix,iz] * Ex[ix,iz]
        oldPz2[iq,ix,iz] = oldPz1[iq,ix,iz]
        oldPz1[iq,ix,iz] = Pz[iq,ix,iz]
        Pz[iq,ix,iz] = Aq[iq,ix,iz] * Pz[iq,ix,iz] +
                       Bq[iq,ix,iz] * oldPz2[iq,ix,iz] +
                       Cq[iq,ix,iz] * Ez[ix,iz]
    end
    return nothing
end
function update_P!(model::Model2D{F,S,T,R,A,AP,V}) where {F,S,T,R,A<:CuArray,AP,V}
    (; Px) = model
    N = length(Px)

    # @krun N update_P_kernel!(model)

    # Have to pass specific field since Fcomp in source is Symbol and not isbits
    (; field, Aq, Bq, Cq, Px, oldPx1, oldPx2, Pz, oldPz1, oldPz2) = model
    @krun N update_P_kernel!(field, Aq, Bq, Cq, Px, oldPx1, oldPx2, Pz, oldPz1, oldPz2)
    return nothing
end
# function update_P_kernel!(model::Model2D)
function update_P_kernel!(field::Field2D, Aq, Bq, Cq, Px, oldPx1, oldPx2, Pz, oldPz1, oldPz2)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    # (; field, Aq, Bq, Cq, Px, oldPx1, oldPx2, Pz, oldPz1, oldPz2) = model
    (; Ex, Ez) = field

    ci = CartesianIndices(Px)
    for ici=id:stride:length(ci)
        iq = ci[ici][1]
        ix = ci[ici][2]
        iz = ci[ici][3]
        oldPx2[iq,ix,iz] = oldPx1[iq,ix,iz]
        oldPx1[iq,ix,iz] = Px[iq,ix,iz]
        Px[iq,ix,iz] = Aq[iq,ix,iz] * Px[iq,ix,iz] +
                       Bq[iq,ix,iz] * oldPx2[iq,ix,iz] +
                       Cq[iq,ix,iz] * Ex[ix,iz]
        oldPz2[iq,ix,iz] = oldPz1[iq,ix,iz]
        oldPz1[iq,ix,iz] = Pz[iq,ix,iz]
        Pz[iq,ix,iz] = Aq[iq,ix,iz] * Pz[iq,ix,iz] +
                       Bq[iq,ix,iz] * oldPz2[iq,ix,iz] +
                       Cq[iq,ix,iz] * Ez[ix,iz]
    end
    return nothing
end


function update_E!(model::Model2D)
    (; field, Me, Sx, Sz, Px, Pz) = model
    (; Ex, Ez, Dx, Dz) = field
    Nq, Nx, Nz = size(Px)
    for iz=1:Nz, ix=1:Nx
        sumPx = zero(eltype(Px))
        sumPz = zero(eltype(Pz))
        for iq=1:Nq
            sumPx += Px[iq,ix,iz]
            sumPz += Pz[iq,ix,iz]
        end
        Ex[ix,iz] = Me[ix,iz] * (Dx[ix,iz] - Sx[ix,iz] - sumPx)
        Ez[ix,iz] = Me[ix,iz] * (Dz[ix,iz] - Sz[ix,iz] - sumPz)
    end
    return nothing
end
function update_E!(model::Model2D{F,S,T,R,A,AP,V}) where {F,S,T,R,A<:CuArray,AP,V}
    (; Px) = model
    Nq, Nx, Nz = size(Px)

    # @krun Nx*Nz update_E_kernel!(model)

    # Have to pass specific field since Fcomp in source is Symbol and not isbits
    (; field, Me, Sx, Sz, Px, Pz) = model
    @krun Nx*Nz update_E_kernel!(field, Me, Sx, Sz, Px, Pz)

    return nothing
end
# function update_E_kernel!(model::Model2D)
function update_E_kernel!(field::Field2D, Me, Sx, Sz, Px, Pz)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    # (; field, Me, Sx, Sz, Px, Pz) = model
    (; Ex, Ez, Dx, Dz) = field

    Nq = size(Px, 1)
    ci = CartesianIndices(Ex)
    for ici=id:stride:length(ci)
        ix = ci[ici][1]
        iz = ci[ici][2]
        sumPx = zero(eltype(Px))
        sumPz = zero(eltype(Pz))
        for iq=1:Nq
            sumPx += Px[iq,ix,iz]
            sumPz += Pz[iq,ix,iz]
        end
        Ex[ix,iz] = Me[ix,iz] * (Dx[ix,iz] - Sx[ix,iz] - sumPx)
        Ez[ix,iz] = Me[ix,iz] * (Dz[ix,iz] - Sz[ix,iz] - sumPz)
    end
    return nothing
end


function update_E_explicit!(model::Model2D)
    (; field, dt, Me, Kx, Kz, psiHyx, psiHyz) = model
    (; Ex, Ez, dHyx, dHyz) = field
    @. Ex = Ex + dt * Me * ((0 - dHyz) + (0 - psiHyz))
    @. Ez = Ez + dt * Me * ((dHyx - 0) + (psiHyx - 0))
    return nothing
end


function update_S!(model::Model2D)
    (; field, Ms, Sx, Sz) = model
    (; Ex, Ez) = field
    @. Sx = Sx + Ms * Ex
    @. Sz = Sz + Ms * Ez
    return nothing
end


# ******************************************************************************
# 3D
# ******************************************************************************
struct Model3D{F, S, T, R, A, AP, V}
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
    Sy :: A
    Sz :: A
    # Variables for ADE dispersion calculation:
    Aq :: AP
    Bq :: AP
    Cq :: AP
    Px :: AP
    oldPx1 :: AP
    oldPx2 :: AP
    Py :: AP
    oldPy1 :: AP
    oldPy2 :: AP
    Pz :: AP
    oldPz1 :: AP
    oldPz2 :: AP
    # CPML variables:
    Kx :: V
    Ax :: V
    Bx :: V
    Ky :: V
    Ay :: V
    By :: V
    Kz :: V
    Az :: V
    Bz :: V
    psiExy :: A
    psiExz :: A
    psiEyx :: A
    psiEyz :: A
    psiEzx :: A
    psiEzy :: A
    psiHxy :: A
    psiHxz :: A
    psiHyx :: A
    psiHyz :: A
    psiHzx :: A
    psiHzy :: A
end

@adapt_structure Model3D


function Model(
    field::Field3D, source;
    tmax,
    CN=1,
    geometry,
    material,
    pml_box=(0,0,0,0,0,0),
)
    (; grid) = field
    (; Nx, Ny, Nz, dx, dy, dz, x, y, z) = grid

    # Time grid:
    dt = CN / C0 / sqrt(1/dx^2 + 1/dy^2 + 1/dz^2)
    Nt = ceil(Int, tmax / dt)
    t = range(0, tmax, Nt)

    # Permittivity, permeability, and conductivity:
    (; eps, mu, sigma) = material
    eps = [geometry(x[ix],y[iy],z[iz]) ? eps : 1 for ix=1:Nx, iy=1:Ny, iz=1:Nz]
    mu = [geometry(x[ix],y[iy],z[iz]) ? mu : 1 for ix=1:Nx, iy=1:Ny, iz=1:Nz]
    sigma = [geometry(x[ix],y[iy],z[iz]) ? sigma : 0 for ix=1:Nx, iy=1:Ny, iz=1:Nz]

    # Update coefficients for H and E fields:
    Mh = @. dt / (MU0*mu)
    Me = @. 1 / (EPS0*eps + sigma*dt)

    # Variables for conductivity calculation:
    Ms = @. sigma * dt
    Sx = zeros(Nx,Ny,Nz)
    Sy = zeros(Nx,Ny,Nz)
    Sz = zeros(Nx,Ny,Nz)

    # Variables for ADE dispersion calculation:
    (; chi) = material
    Nq = length(chi)
    Aq, Bq, Cq = (zeros(Nq,Nx,Ny,Nz) for i=1:3)
    for iz=1:Nz, iy=1:Ny, ix=1:Nx, iq=1:Nq
        Aq0, Bq0, Cq0 = ade_coefficients(chi[iq], dt)
        Aq[iq,ix,iy,iz] = geometry(x[ix],y[iy],z[iz]) * Aq0
        Bq[iq,ix,iy,iz] = geometry(x[ix],y[iy],z[iz]) * Bq0
        Cq[iq,ix,iy,iz] = geometry(x[ix],y[iy],z[iz]) * Cq0
    end
    Px, oldPx1, oldPx2 = (zeros(Nq,Nx,Ny,Nz) for i=1:3)
    Py, oldPy1, oldPy2 = (zeros(Nq,Nx,Ny,Nz) for i=1:3)
    Pz, oldPz1, oldPz2 = (zeros(Nq,Nx,Ny,Nz) for i=1:3)

    # CPML variables:
    Kx, Ax, Bx = pml(x, pml_box[1:2], dt)
    Ky, Ay, By = pml(y, pml_box[3:4], dt)
    Kz, Az, Bz = pml(z, pml_box[5:6], dt)
    psiExy, psiExz, psiEyx, psiEyz, psiEzx, psiEzy = (zeros(Nx,Ny,Nz) for i=1:6)
    psiHxy, psiHxz, psiHyx, psiHyz, psiHzx, psiHzy = (zeros(Nx,Ny,Nz) for i=1:6)

    return Model3D(
        field, source, Nt, dt, t, Mh, Me, Ms, Sx, Sy, Sz,
        Aq, Bq, Cq, Px, oldPx1, oldPx2, Py, oldPy1, oldPy2, Pz, oldPz1, oldPz2,
        Kx, Ax, Bx, Ky, Ay, By, Kz, Az, Bz,
        psiExy, psiExz, psiEyx, psiEyz, psiEzx, psiEzy,
        psiHxy, psiHxz, psiHyx, psiHyz, psiHzx, psiHzy,
    )
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


function update_D!(model::Model3D)
    (; field, dt, Kx, Ky, Kz) = model
    (; psiHxy, psiHxz, psiHyx, psiHyz, psiHzx, psiHzy) = model
    (; Dx, Dy, Dz, dHxy, dHxz, dHyx, dHyz, dHzx, dHzy) = field
    @. Dx = Dx + dt * ((dHzy - dHyz) + (psiHzy - psiHyz))
    @. Dy = Dy + dt * ((dHxz - dHzx) + (psiHxz - psiHzx))
    @. Dz = Dz + dt * ((dHyx - dHxy) + (psiHyx - psiHxy))
    return nothing
end


function update_P!(model::Model3D)
    (; field, Aq, Bq, Cq) = model
    (; Px, oldPx1, oldPx2, Py, oldPy1, oldPy2, Pz, oldPz1, oldPz2) = model
    (; Ex, Ey, Ez) = field
    Nq, Nx, Ny, Nz = size(Px)
    for iz=1:Nz, iy=1:Ny, ix=1:Nx, iq=1:Nq
        oldPx2[iq,ix,iy,iz] = oldPx1[iq,ix,iy,iz]
        oldPx1[iq,ix,iy,iz] = Px[iq,ix,iy,iz]
        Px[iq,ix,iy,iz] = Aq[iq,ix,iy,iz] * Px[iq,ix,iy,iz] +
                          Bq[iq,ix,iy,iz] * oldPx2[iq,ix,iy,iz] +
                          Cq[iq,ix,iy,iz] * Ex[ix,iy,iz]
        oldPy2[iq,ix,iy,iz] = oldPy1[iq,ix,iy,iz]
        oldPy1[iq,ix,iy,iz] = Py[iq,ix,iy,iz]
        Py[iq,ix,iy,iz] = Aq[iq,ix,iy,iz] * Py[iq,ix,iy,iz] +
                          Bq[iq,ix,iy,iz] * oldPy2[iq,ix,iy,iz] +
                          Cq[iq,ix,iy,iz] * Ey[ix,iy,iz]
        oldPz2[iq,ix,iy,iz] = oldPz1[iq,ix,iy,iz]
        oldPz1[iq,ix,iy,iz] = Pz[iq,ix,iy,iz]
        Pz[iq,ix,iy,iz] = Aq[iq,ix,iy,iz] * Pz[iq,ix,iy,iz] +
                          Bq[iq,ix,iy,iz] * oldPz2[iq,ix,iy,iz] +
                          Cq[iq,ix,iy,iz] * Ez[ix,iy,iz]
    end
    return nothing
end
function update_P!(model::Model3D{F,S,T,R,A,AP,V}) where {F,S,T,R,A<:CuArray,AP,V}
    (; Px) = model
    N = length(Px)

    # @krun N update_P_kernel!(model)

    # Have to pass specific field since Fcomp in source is Symbol and not isbits
    (; field, Aq, Bq, Cq) = model
    (; Px, oldPx1, oldPx2, Py, oldPy1, oldPy2, Pz, oldPz1, oldPz2) = model
    @krun N update_P_kernel!(field, Aq, Bq, Cq, Px, oldPx1, oldPx2, Py, oldPy1, oldPy2, Pz, oldPz1, oldPz2)
    return nothing
end
# function update_P_kernel!(model::Model3D)
function update_P_kernel!(
    field::Field3D, Aq, Bq, Cq,
    Px, oldPx1, oldPx2, Py, oldPy1, oldPy2, Pz, oldPz1, oldPz2,
)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    (; Ex, Ey, Ez) = field
    ci = CartesianIndices(Px)
    for ici=id:stride:length(ci)
        iq = ci[ici][1]
        ix = ci[ici][2]
        iy = ci[ici][3]
        iz = ci[ici][4]
        oldPx2[iq,ix,iy,iz] = oldPx1[iq,ix,iy,iz]
        oldPx1[iq,ix,iy,iz] = Px[iq,ix,iy,iz]
        Px[iq,ix,iy,iz] = Aq[iq,ix,iy,iz] * Px[iq,ix,iy,iz] +
                          Bq[iq,ix,iy,iz] * oldPx2[iq,ix,iy,iz] +
                          Cq[iq,ix,iy,iz] * Ex[ix,iy,iz]
        oldPy2[iq,ix,iy,iz] = oldPy1[iq,ix,iy,iz]
        oldPy1[iq,ix,iy,iz] = Py[iq,ix,iy,iz]
        Py[iq,ix,iy,iz] = Aq[iq,ix,iy,iz] * Py[iq,ix,iy,iz] +
                          Bq[iq,ix,iy,iz] * oldPy2[iq,ix,iy,iz] +
                          Cq[iq,ix,iy,iz] * Ey[ix,iy,iz]
        oldPz2[iq,ix,iy,iz] = oldPz1[iq,ix,iy,iz]
        oldPz1[iq,ix,iy,iz] = Pz[iq,ix,iy,iz]
        Pz[iq,ix,iy,iz] = Aq[iq,ix,iy,iz] * Pz[iq,ix,iy,iz] +
                          Bq[iq,ix,iy,iz] * oldPz2[iq,ix,iy,iz] +
                          Cq[iq,ix,iy,iz] * Ez[ix,iy,iz]
    end
    return nothing
end


function update_E!(model::Model3D)
    (; field, Me, Sx, Sy, Sz, Px, Py, Pz) = model
    (; Ex, Ey, Ez, Dx, Dy, Dz) = field
    Nq, Nx, Ny, Nz = size(Px)
    for iz=1:Nz, iy=1:Ny, ix=1:Nx
        sumPx = zero(eltype(Px))
        sumPy = zero(eltype(Py))
        sumPz = zero(eltype(Pz))
        for iq=1:Nq
            sumPx += Px[iq,ix,iy,iz]
            sumPy += Py[iq,ix,iy,iz]
            sumPz += Pz[iq,ix,iy,iz]
        end
        Ex[ix,iy,iz] = Me[ix,iy,iz] * (Dx[ix,iy,iz] - Sx[ix,iy,iz] - sumPx)
        Ey[ix,iy,iz] = Me[ix,iy,iz] * (Dy[ix,iy,iz] - Sy[ix,iy,iz] - sumPy)
        Ez[ix,iy,iz] = Me[ix,iy,iz] * (Dz[ix,iy,iz] - Sz[ix,iy,iz] - sumPz)
    end
    return nothing
end
function update_E!(model::Model3D{F,S,T,R,A,AP,V}) where {F,S,T,R,A<:CuArray,AP,V}
    (; Px) = model
    Nq, Nx, Ny, Nz = size(Px)

    # @krun Nx*Ny*Nz update_E_kernel!(model)

    # Have to pass specific field since Fcomp in source is Symbol and not isbits
    (; field, Me, Sx, Sy, Sz, Px, Py, Pz) = model
    @krun Nx*Ny*Nz update_E_kernel!(field, Me, Sx, Sy, Sz, Px, Py, Pz)

    return nothing
end
# function update_E_kernel!(model::Model3D)
function update_E_kernel!(field::Field3D, Me, Sx, Sy, Sz, Px, Py, Pz)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    (; Ex, Ey, Ez, Dx, Dy, Dz) = field
    Nq = size(Px, 1)
    ci = CartesianIndices(Ex)
    for ici=id:stride:length(ci)
        ix = ci[ici][1]
        iy = ci[ici][2]
        iz = ci[ici][3]
        sumPx = zero(eltype(Px))
        sumPy = zero(eltype(Py))
        sumPz = zero(eltype(Pz))
        for iq=1:Nq
            sumPx += Px[iq,ix,iy,iz]
            sumPy += Py[iq,ix,iy,iz]
            sumPz += Pz[iq,ix,iy,iz]
        end
        Ex[ix,iy,iz] = Me[ix,iy,iz] * (Dx[ix,iy,iz] - Sx[ix,iy,iz] - sumPx)
        Ey[ix,iy,iz] = Me[ix,iy,iz] * (Dy[ix,iy,iz] - Sy[ix,iy,iz] - sumPy)
        Ez[ix,iy,iz] = Me[ix,iy,iz] * (Dz[ix,iy,iz] - Sz[ix,iy,iz] - sumPz)
    end
    return nothing
end


function update_E_explicit!(model::Model3D)
    (; field, dt, Me, Kx, Ky, Kz) = model
    (; psiHxy, psiHxz, psiHyx, psiHyz, psiHzx, psiHzy) = model
    (; Ex, Ey, Ez, dHxy, dHxz, dHyx, dHyz, dHzx, dHzy) = field
    @. Ex = Ex + dt * Me * ((dHzy - dHyz) + (psiHzy - psiHyz))
    @. Ey = Ey + dt * Me * ((dHxz - dHzx) + (psiHxz - psiHzx))
    @. Ez = Ez + dt * Me * ((dHyx - dHxy) + (psiHyx - psiHxy))
    return nothing
end


function update_S!(model::Model3D)
    (; field, Ms, Sx, Sy, Sz) = model
    (; Ex, Ey, Ez) = field
    @. Sx = Sx + Ms * Ex
    @. Sy = Sy + Ms * Ey
    @. Sz = Sz + Ms * Ez
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
