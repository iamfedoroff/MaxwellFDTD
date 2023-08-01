abstract type Model end


function step!(model, it)
    update_H!(model)
    update_E!(model)
    add_source!(model, model.source, it)
    return nothing
end


function solve!(
    model; arch=CPU(), fname=nothing, nstride=nothing, nframes=nothing, dtout=nothing,
    tfsf_record=false, tfsf_box=nothing, tfsf_fname=nothing,
)
    model = adapt(arch, model)
    (; Nt, dt, t) = model

    if isnothing(fname)
        fname = default_fname(model)
    end

    if tfsf_record
        if isnothing(tfsf_fname)
            ext = splitext(fname)[end]
            tfsf_fname = replace(fname, ext => "_tfsf" * ext)
        end
        if !isdir(dirname(tfsf_fname))
            mkpath(dirname(tfsf_fname))
        end
        tfsf_data = prepare_tfsf_record(model, tfsf_box, tfsf_fname)
    end

    out = Output(model; fname, nstride, nframes, dtout)

    @showprogress 1 for it=1:Nt
        @timeit "model step" begin
            step!(model, it)
            if CUDA.functional()
                synchronize()
            end
        end

        @timeit "output" begin
            if (out.itout <= out.Ntout) && (abs(t[it] - out.tout[out.itout]) <= dt/2)
                write_output!(out, model)
                out.itout += 1
            end
            update_output_variables(out, model)
            if CUDA.functional()
                synchronize()
            end
        end

        if tfsf_record
            write_tfsf_record(model, tfsf_data, it)
        end

        if it == 1
            reset_timer!()
        end
    end

    write_output_values(out, model)

    print_timer()

    return nothing
end


# ******************************************************************************************
# 1D: d/dx = d/dy = 0,   (Hy, Ex)
# ******************************************************************************************
struct Model1D{F, S, P, T, R, A, AP}
    field :: F
    source :: S
    pml :: P
    # Time grid:
    Nt :: Int
    dt :: T
    t :: R
    # Update coefficients for H, E and D fields:
    Mh :: A
    Me :: A
    Md1 :: A
    Md2 :: A
    # Variables for ADE dispersion calculation:
    Aq :: AP
    Bq :: AP
    Cq :: AP
    Px :: AP
    oldPx1 :: AP
    oldPx2 :: AP
end

@adapt_structure Model1D


function Model(
    grid::Grid1D, source_data; tmax, CN=0.5, material=nothing, pml_box=(0,0),
)
    (; Nz, dz, z) = grid

    field = Field(grid)

    # Time grid:
    dt = CN / C0 / sqrt(1/dz^2)
    Nt = ceil(Int, tmax / dt)
    t = range(0, tmax, Nt)

    source = source_init(source_data, field, t)

    pml = PML(z, pml_box, dt)

    # Material processing:
    if isnothing(material)
        material = Material(geometry = z -> false)
    end
    (; geometry, eps, mu, sigma, chi) = material

    # Permittivity, permeability, and conductivity:
    eps = [geometry(z[iz]) ? eps : 1 for iz=1:Nz]
    mu = [geometry(z[iz]) ? mu : 1 for iz=1:Nz]
    sigma = [geometry(z[iz]) ? sigma : 0 for iz=1:Nz]
    @. sigma = sigma / (EPS0*eps)   # J=sigma*E -> J=sigma*D

    # Variables for ADE dispersion calculation:
    Nq = length(chi)
    Aq, Bq, Cq = (zeros(Nq,Nz) for i=1:3)
    for iz=1:Nz, iq=1:Nq
        Aq0, Bq0, Cq0 = ade_coefficients(chi[iq], dt)
        Aq[iq,iz] = geometry(z[iz]) * Aq0
        Bq[iq,iz] = geometry(z[iz]) * Bq0
        Cq[iq,iz] = geometry(z[iz]) * Cq0
    end
    Px, oldPx1, oldPx2 = (zeros(Nq,Nz) for i=1:3)


    # Compensation for the numerical dispersion:
    # dt = t[2] - t[1]
    # lam0 = 2e-6   # (m) wavelength
    # w0 = 2*pi * C0 / lam0   # frequency
    # sn = C0 * dt/dz * sin(w0/C0 * dz/2) / sin(w0 * dt/2)
    # @show sn
    # eps = @. sn * eps
    # mu = @. sn * mu


    # Update coefficients for H, E and D fields:
    Mh = @. dt / (MU0*mu)
    Me = @. 1 / (EPS0*eps)
    Md1 = @. (1 - sigma*dt/2) / (1 + sigma*dt/2)
    Md2 = @. dt / (1 + sigma*dt/2)

    return Model1D(
        field, source, pml, Nt, dt, t, Mh, Me, Md1, Md2,
        Aq, Bq, Cq, Px, oldPx1, oldPx2,
    )
end


@kernel function update_H_kernel!(model::Model1D)
    (; field, pml, Mh) = model
    (; zlayer1, psiExz1, zlayer2, psiExz2) = pml
    (; grid, Hy, Ex) = field
    (; Nz, dz) = grid

    iz = @index(Global)

    @inbounds begin
        # derivatives E:
        iz == Nz ? izp1 = 1 : izp1 = iz + 1
        dExz = (Ex[izp1] - Ex[iz]) / dz

        # apply CPML:
        if iz <= zlayer1.ind   # z left layer [1:iz1]
            (; K, A, B) = zlayer1
            izpml = iz
            psiExz1[izpml] = B[izpml] * psiExz1[izpml] + A[izpml] * dExz
            Hy[iz] -= Mh[iz] * psiExz1[izpml]
            dExz = dExz / K[izpml]
        end
        if iz >= zlayer2.ind   # z right layer [iz1:Nz]
            (; ind, K, A, B) = zlayer2
            izpml = iz - ind + 1
            psiExz2[izpml] = B[izpml] * psiExz2[izpml] + A[izpml] * dExz
            Hy[iz] -= Mh[iz] * psiExz2[izpml]
            dExz = dExz / K[izpml]
        end

        # update H:
        Hy[iz] -= Mh[iz] * (0 + dExz)
    end
end
function update_H!(model::Model1D)
    (; Hy) = model.field
    backend = get_backend(Hy)
    ndrange = size(Hy)
    update_H_kernel!(backend)(model; ndrange)
    return nothing
end


@kernel function update_E_kernel!(model::Model1D)
    (; field, pml, Me, Md1, Md2) = model
    (; Aq, Bq, Cq, Px, oldPx1, oldPx2) = model
    (; zlayer1, psiHyz1, zlayer2, psiHyz2) = pml
    (; grid, Hy, Dx, Ex) = field
    (; Nz, dz) = grid

    iz = @index(Global)

    @inbounds begin
        # derivatives H:
        iz == 1 ? izm1 = Nz : izm1 = iz - 1
        dHyz = (Hy[iz] - Hy[izm1]) / dz

        # apply CPML:
        if iz <= zlayer1.ind   # z left layer [1:iz1]
            (; K, A, B) = zlayer1
            izpml = iz
            psiHyz1[izpml] = B[izpml] * psiHyz1[izpml] + A[izpml] * dHyz
            Dx[iz] -= Md2[iz] * psiHyz1[izpml]
            dHyz = dHyz / K[izpml]
        end
        if iz >= zlayer2.ind   # z right layer [iz1:Nz]
            (; ind, K, A, B) = zlayer2
            izpml = iz - ind + 1
            psiHyz2[izpml] = B[izpml] * psiHyz2[izpml] + A[izpml] * dHyz
            Dx[iz] -= Md2[iz] * psiHyz2[izpml]
            dHyz = dHyz / K[izpml]
        end

        # update D:
        Dx[iz] = Md1[iz] * Dx[iz] + Md2[iz] * (0 - dHyz)

        # update P:
        Nq = size(Px, 1)
        sumPx = zero(eltype(Px))
        for iq=1:Nq
            oldPx2[iq,iz] = oldPx1[iq,iz]
            oldPx1[iq,iz] = Px[iq,iz]
            Px[iq,iz] = Aq[iq,iz] * Px[iq,iz] +
                        Bq[iq,iz] * oldPx2[iq,iz] +
                        Cq[iq,iz] * Ex[iz]
            sumPx += Px[iq,iz]
        end

        # update E:
        Ex[iz] = Me[iz] * (Dx[iz] - sumPx)


        # update E explicit:
        # (; dt) = model
        # Ex[iz] += dt * Me[iz] * ((0 - dHyz / Kz[iz]) + (0 - psiHyz[iz]))
    end
end
function update_E!(model::Model1D)
    (; Ex) = model.field
    backend = get_backend(Ex)
    ndrange = size(Ex)
    update_E_kernel!(backend)(model; ndrange)
    return nothing
end


# ******************************************************************************************
# 2D
# ******************************************************************************************
struct Model2D{F, S, P, T, R, A, AP}
    field :: F
    source :: S
    pml :: P
    # Time grid:
    Nt :: Int
    dt :: T
    t :: R
    # Update coefficients for H, E and D fields:
    Mh :: A
    Me :: A
    Md1 :: A
    Md2 :: A
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
end

@adapt_structure Model2D


function Model(
    grid::Grid2D, source_data; tmax, CN=0.5, material=nothing, pml_box=(0,0,0,0),
)
    (; Nx, Nz, dx, dz, x, z) = grid

    field = Field(grid)

    # Time grid:
    dt = CN / C0 / sqrt(1/dx^2 + 1/dz^2)
    Nt = ceil(Int, tmax / dt)
    t = range(0, tmax, Nt)

    source = source_init(source_data, field, t)

    pml = PML(x, z, pml_box, dt)

    # Material processing:
    if isnothing(material)
        material = Material(geometry = (x,z) -> false)
    end
    (; geometry, eps, mu, sigma, chi) = material

    # Permittivity, permeability, and conductivity:
    eps = [geometry(x[ix],z[iz]) ? eps : 1 for ix=1:Nx, iz=1:Nz]
    mu = [geometry(x[ix],z[iz]) ? mu : 1 for ix=1:Nx, iz=1:Nz]
    sigma = [geometry(x[ix],z[iz]) ? sigma : 0 for ix=1:Nx, iz=1:Nz]
    @. sigma = sigma / (EPS0*eps)   # J=sigma*E -> J=sigma*D

    # Variables for ADE dispersion calculation:
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

    # Update coefficients for H, E and D fields:
    Mh = @. dt / (MU0*mu)
    Me = @. 1 / (EPS0*eps)
    Md1 = @. (1 - sigma*dt/2) / (1 + sigma*dt/2)
    Md2 = @. dt / (1 + sigma*dt/2)

    return Model2D(
        field, source, pml, Nt, dt, t, Mh, Me, Md1, Md2,
        Aq, Bq, Cq, Px, oldPx1, oldPx2, Pz, oldPz1, oldPz2,
    )
end


@kernel function update_H_kernel!(model::Model2D)
    (; field, pml, Mh) = model
    (; xlayer1, psiEzx1, xlayer2, psiEzx2, zlayer1, psiExz1, zlayer2, psiExz2) = pml
    (; grid, Hy, Ex, Ez) = field
    (; Nx, Nz, dx, dz) = grid

    ix, iz = @index(Global, NTuple)

    @inbounds begin
        # derivatives E:
        ix == Nx ? ixp1 = 1 : ixp1 = ix + 1
        iz == Nz ? izp1 = 1 : izp1 = iz + 1
        dExz = (Ex[ix,izp1] - Ex[ix,iz]) / dz
        dEzx = (Ez[ixp1,iz] - Ez[ix,iz]) / dx

        # apply CPML:
        if ix <= xlayer1.ind   # x left layer [1:ix1]
            (; K, A, B) = xlayer1
            ixpml = ix
            psiEzx1[ixpml,iz] = B[ixpml] * psiEzx1[ixpml,iz] + A[ixpml] * dEzx
            Hy[ix,iz] += Mh[ix,iz] * psiEzx1[ixpml,iz]
            dEzx = dEzx / K[ixpml]
        end
        if ix >= xlayer2.ind      # x right layer [ix2:Nx]
            (; ind, K, A, B) = xlayer2
            ixpml = ix - ind + 1
            psiEzx2[ixpml,iz] = B[ixpml] * psiEzx2[ixpml,iz] + A[ixpml] * dEzx
            Hy[ix,iz] += Mh[ix,iz] * psiEzx2[ixpml,iz]
            dEzx = dEzx / K[ixpml]
        end
        if iz <= zlayer1.ind   # z left layer [1:iz1]
            (; K, A, B) = zlayer1
            izpml = iz
            psiExz1[ix,izpml] = B[izpml] * psiExz1[ix,izpml] + A[izpml] * dExz
            Hy[ix,iz] -= Mh[ix,iz] * psiExz1[ix,izpml]
            dExz = dExz / K[izpml]
        end
        if iz >= zlayer2.ind      # z right layer [iz1:Nz]
            (; ind, K, A, B) = zlayer2
            izpml = iz - ind + 1
            psiExz2[ix,izpml] = B[izpml] * psiExz2[ix,izpml] + A[izpml] * dExz
            Hy[ix,iz] -= Mh[ix,iz] * psiExz2[ix,izpml]
            dExz = dExz / K[izpml]
        end

        # update H:
        Hy[ix,iz] -= Mh[ix,iz] * (dExz - dEzx)
    end
end
function update_H!(model::Model2D)
    (; Hy) = model.field
    backend = get_backend(Hy)
    ndrange = size(Hy)
    update_H_kernel!(backend)(model; ndrange)
    return nothing
end


@kernel function update_E_kernel!(model::Model2D)
    (; field, pml, Me, Md1, Md2) = model
    (; Aq, Bq, Cq, Px, oldPx1, oldPx2, Pz, oldPz1, oldPz2) = model
    (; xlayer1, psiHyx1, xlayer2, psiHyx2, zlayer1, psiHyz1, zlayer2, psiHyz2) = pml
    (; grid, Hy, Dx, Dz, Ex, Ez) = field
    (; Nx, Nz, dx, dz) = grid

    ix, iz = @index(Global, NTuple)

    @inbounds begin
        # derivatives H:
        ix == 1 ? ixm1 = Nx : ixm1 = ix - 1
        iz == 1 ? izm1 = Nz : izm1 = iz - 1
        dHyx = (Hy[ix,iz] - Hy[ixm1,iz]) / dx
        dHyz = (Hy[ix,iz] - Hy[ix,izm1]) / dz

        # apply CPML:
        if ix <= xlayer1.ind   # x left layer [1:ix1]
            (; K, A, B) = xlayer1
            ixpml = ix
            psiHyx1[ixpml,iz] = B[ixpml] * psiHyx1[ixpml,iz] + A[ixpml] * dHyx
            Dz[ix,iz] += Md2[ix,iz] * psiHyx1[ixpml,iz]
            dHyx = dHyx / K[ixpml]
        end
        if ix >= xlayer2.ind   # x right layer [ix2:Nx]
            (; ind, K, A, B) = xlayer2
            ixpml = ix - ind + 1
            psiHyx2[ixpml,iz] = B[ixpml] * psiHyx2[ixpml,iz] + A[ixpml] * dHyx
            Dz[ix,iz] += Md2[ix,iz] * psiHyx2[ixpml,iz]
            dHyx = dHyx / K[ixpml]
        end
        if iz <= zlayer1.ind   # z left layer [1:iz1]
            (; K, A, B) = zlayer1
            izpml = iz
            psiHyz1[ix,izpml] = B[izpml] * psiHyz1[ix,izpml] + A[izpml] * dHyz
            Dx[ix,iz] -= Md2[ix,iz] * psiHyz1[ix,izpml]
            dHyz = dHyz / K[izpml]
        end
        if iz >= zlayer2.ind   # z right layer [iz2:Nz]
            (; ind, K, A, B) = zlayer2
            izpml = iz - ind + 1
            psiHyz2[ix,izpml] = B[izpml] * psiHyz2[ix,izpml] + A[izpml] * dHyz
            Dx[ix,iz] -= Md2[ix,iz] * psiHyz2[ix,izpml]
            dHyz = dHyz / K[izpml]
        end

        # update D:
        Dx[ix,iz] = Md1[ix,iz] * Dx[ix,iz] + Md2[ix,iz] * (0 - dHyz)
        Dz[ix,iz] = Md1[ix,iz] * Dz[ix,iz] + Md2[ix,iz] * (dHyx - 0)

        # update P:
        Nq = size(Px, 1)
        sumPx = zero(eltype(Px))
        sumPz = zero(eltype(Pz))
        for iq=1:Nq
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
            sumPx += Px[iq,ix,iz]
            sumPz += Pz[iq,ix,iz]
        end

        # update E:
        Ex[ix,iz] = Me[ix,iz] * (Dx[ix,iz] - sumPx)
        Ez[ix,iz] = Me[ix,iz] * (Dz[ix,iz] - sumPz)


        # update E explicit:
        # (; dt) = model
        # Ex[ix,iz] += dt * Me[ix,iz] * ((0 - dHyz / Kz[iz]) + (0 - psiHyz[ix,iz]))
        # Ez[ix,iz] += dt * Me[ix,iz] * ((dHyx / Kx[ix] - 0) + (psiHyx[ix,iz] - 0))
    end
end
function update_E!(model::Model2D)
    (; Ex) = model.field
    backend = get_backend(Ex)
    ndrange = size(Ex)
    update_E_kernel!(backend)(model; ndrange)
    return nothing
end


# ******************************************************************************************
# 3D
# ******************************************************************************************
struct Model3D{F, S, P, T, R, A, AP}
    field :: F
    source :: S
    pml :: P
    # Time grid:
    Nt :: Int
    dt :: T
    t :: R
    # Update coefficients for H, E and D fields:
    Mh :: A
    Me :: A
    Md1 :: A
    Md2 :: A
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
end

@adapt_structure Model3D


function Model(
    grid::Grid3D, source_data; tmax, CN=0.5, material=nothing, pml_box=(0,0,0,0,0,0),
)
    (; Nx, Ny, Nz, dx, dy, dz, x, y, z) = grid

    field = Field(grid)

    # Time grid:
    dt = CN / C0 / sqrt(1/dx^2 + 1/dy^2 + 1/dz^2)
    Nt = ceil(Int, tmax / dt)
    t = range(0, tmax, Nt)

    source = source_init(source_data, field, t)

    pml = PML(x, y, z, pml_box, dt)

    # Material processing:
    if isnothing(material)
        material = Material(geometry = (x,y,z) -> false)
        eps = 1
        mu = 1
        sigma = 0
        chi = [nothing]
    end
    (; geometry, eps, mu, sigma, chi) = material

    # Permittivity, permeability, and conductivity:
    eps = [geometry(x[ix],y[iy],z[iz]) ? eps : 1 for ix=1:Nx, iy=1:Ny, iz=1:Nz]
    mu = [geometry(x[ix],y[iy],z[iz]) ? mu : 1 for ix=1:Nx, iy=1:Ny, iz=1:Nz]
    sigma = [geometry(x[ix],y[iy],z[iz]) ? sigma : 0 for ix=1:Nx, iy=1:Ny, iz=1:Nz]
    @. sigma = sigma / (EPS0*eps)   # J=sigma*E -> J=sigma*D

    # Variables for ADE dispersion calculation:
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

    # Update coefficients for H, E and D fields:
    Mh = @. dt / (MU0*mu)
    Me = @. 1 / (EPS0*eps)
    Md1 = @. (1 - sigma*dt/2) / (1 + sigma*dt/2)
    Md2 = @. dt / (1 + sigma*dt/2)

    return Model3D(
        field, source, pml, Nt, dt, t, Mh, Me, Md1, Md2,
        Aq, Bq, Cq, Px, oldPx1, oldPx2, Py, oldPy1, oldPy2, Pz, oldPz1, oldPz2,
    )
end


# In order to avoid the issue caused by the large size of the CUDA kernel parameters,
# here we pass the parameters of the model explicitly:
# https://discourse.julialang.org/t/passing-too-long-tuples-into-cuda-kernel-causes-an-error
@kernel function update_H_kernel!(field::Field3D, pml, Mh)
    (; xlayer1, psiEyx1, psiEzx1, xlayer2, psiEyx2, psiEzx2,
       ylayer1, psiExy1, psiEzy1, ylayer2, psiExy2, psiEzy2,
       zlayer1, psiExz1, psiEyz1, zlayer2, psiExz2, psiEyz2) = pml
    (; grid, Hx, Hy, Hz, Ex, Ey, Ez) = field
    (; Nx, Ny, Nz, dx, dy, dz) = grid

    ix, iy, iz = @index(Global, NTuple)

    @inbounds begin
        # derivatives E:
        ix == Nx ? ixp1 = 1 : ixp1 = ix + 1
        iy == Ny ? iyp1 = 1 : iyp1 = iy + 1
        iz == Nz ? izp1 = 1 : izp1 = iz + 1
        dExy = (Ex[ix,iyp1,iz] - Ex[ix,iy,iz]) / dy
        dExz = (Ex[ix,iy,izp1] - Ex[ix,iy,iz]) / dz
        dEyx = (Ey[ixp1,iy,iz] - Ey[ix,iy,iz]) / dx
        dEyz = (Ey[ix,iy,izp1] - Ey[ix,iy,iz]) / dz
        dEzx = (Ez[ixp1,iy,iz] - Ez[ix,iy,iz]) / dx
        dEzy = (Ez[ix,iyp1,iz] - Ez[ix,iy,iz]) / dy

        # apply CPML:
        if ix <= xlayer1.ind   # x left layer [1:ix1]
            (; K, A, B) = xlayer1
            ixpml = ix
            psiEyx1[ixpml,iy,iz] = B[ixpml] * psiEyx1[ixpml,iy,iz] + A[ixpml] * dEyx
            psiEzx1[ixpml,iy,iz] = B[ixpml] * psiEzx1[ixpml,iy,iz] + A[ixpml] * dEzx
            Hy[ix,iy,iz] += Mh[ix,iy,iz] * psiEzx1[ixpml,iy,iz]
            Hz[ix,iy,iz] -= Mh[ix,iy,iz] * psiEyx1[ixpml,iy,iz]
            dEyx = dEyx / K[ixpml]
            dEzx = dEzx / K[ixpml]
        end
        if ix >= xlayer2.ind   # x right layer [ix2:Nx]
            (; ind, K, A, B) = xlayer2
            ixpml = ix - ind + 1
            psiEyx2[ixpml,iy,iz] = B[ixpml] * psiEyx2[ixpml,iy,iz] + A[ixpml] * dEyx
            psiEzx2[ixpml,iy,iz] = B[ixpml] * psiEzx2[ixpml,iy,iz] + A[ixpml] * dEzx
            Hy[ix,iy,iz] += Mh[ix,iy,iz] * psiEzx2[ixpml,iy,iz]
            Hz[ix,iy,iz] -= Mh[ix,iy,iz] * psiEyx2[ixpml,iy,iz]
            dEyx = dEyx / K[ixpml]
            dEzx = dEzx / K[ixpml]
        end
        if iy <= ylayer1.ind   # y left layer [1:iy1]
            (; K, A, B) = ylayer1
            iypml = iy
            psiExy1[ix,iypml,iz] = B[iypml] * psiExy1[ix,iypml,iz] + A[iypml] * dExy
            psiEzy1[ix,iypml,iz] = B[iypml] * psiEzy1[ix,iypml,iz] + A[iypml] * dEzy
            Hx[ix,iy,iz] -= Mh[ix,iy,iz] * psiEzy1[ix,iypml,iz]
            Hz[ix,iy,iz] += Mh[ix,iy,iz] * psiExy1[ix,iypml,iz]
            dExy = dExy / K[iypml]
            dEzy = dEzy / K[iypml]
        end
        if iy >= ylayer2.ind   # y right layer [iy2:Ny]
            (; ind, K, A, B) = ylayer2
            iypml = iy - ind + 1
            psiExy2[ix,iypml,iz] = B[iypml] * psiExy2[ix,iypml,iz] + A[iypml] * dExy
            psiEzy2[ix,iypml,iz] = B[iypml] * psiEzy2[ix,iypml,iz] + A[iypml] * dEzy
            Hx[ix,iy,iz] -= Mh[ix,iy,iz] * psiEzy2[ix,iypml,iz]
            Hz[ix,iy,iz] += Mh[ix,iy,iz] * psiExy2[ix,iypml,iz]
            dExy = dExy / K[iypml]
            dEzy = dEzy / K[iypml]
        end
        if iz <= zlayer1.ind   # z left layer [1:iz1]
            (; K, A, B) = zlayer1
            izpml = iz
            psiExz1[ix,iy,izpml] = B[izpml] * psiExz1[ix,iy,izpml] + A[izpml] * dExz
            psiEyz1[ix,iy,izpml] = B[izpml] * psiEyz1[ix,iy,izpml] + A[izpml] * dEyz
            Hx[ix,iy,iz] += Mh[ix,iy,iz] * psiEyz1[ix,iy,izpml]
            Hy[ix,iy,iz] -= Mh[ix,iy,iz] * psiExz1[ix,iy,izpml]
            dExz = dExz / K[izpml]
            dEyz = dEyz / K[izpml]
        end
        if iz >= zlayer2.ind   # z right layer [iz2:Nz]
            (; ind, K, A, B) = zlayer2
            izpml = iz - ind + 1
            psiExz2[ix,iy,izpml] = B[izpml] * psiExz2[ix,iy,izpml] + A[izpml] * dExz
            psiEyz2[ix,iy,izpml] = B[izpml] * psiEyz2[ix,iy,izpml] + A[izpml] * dEyz
            Hx[ix,iy,iz] += Mh[ix,iy,iz] * psiEyz2[ix,iy,izpml]
            Hy[ix,iy,iz] -= Mh[ix,iy,iz] * psiExz2[ix,iy,izpml]
            dExz = dExz / K[izpml]
            dEyz = dEyz / K[izpml]
        end

        # update H:
        Hx[ix,iy,iz] -= Mh[ix,iy,iz] * (dEzy - dEyz)
        Hy[ix,iy,iz] -= Mh[ix,iy,iz] * (dExz - dEzx)
        Hz[ix,iy,iz] -= Mh[ix,iy,iz] * (dEyx - dExy)
    end
end
function update_H!(model::Model3D)
    (; Hx) = model.field
    backend = get_backend(Hx)
    ndrange = size(Hx)
    # In order to avoid the issue caused by the large size of the CUDA kernel parameters,
    # here we pass the parameters of the model explicitly:
    # https://discourse.julialang.org/t/passing-too-long-tuples-into-cuda-kernel-causes-an-error
    (; field, pml, Mh) = model
    update_H_kernel!(backend)(field, pml, Mh; ndrange)
    return nothing
end


# In order to avoid the issue caused by the large size of the CUDA kernel parameters,
# here we pass the parameters of the model explicitly:
# https://discourse.julialang.org/t/passing-too-long-tuples-into-cuda-kernel-causes-an-error
@kernel function update_E_kernel!(
    field::Field3D, pml, Me, Md1, Md2,
    Aq, Bq, Cq, Px, oldPx1, oldPx2, Py, oldPy1, oldPy2, Pz, oldPz1, oldPz2,
)
    (; xlayer1, psiHyx1, psiHzx1, xlayer2, psiHyx2, psiHzx2,
       ylayer1, psiHxy1, psiHzy1, ylayer2, psiHxy2, psiHzy2,
       zlayer1, psiHxz1, psiHyz1, zlayer2, psiHxz2, psiHyz2) = pml
    (; grid, Hx, Hy, Hz, Dx, Dy, Dz, Ex, Ey, Ez) = field
    (; Nx, Ny, Nz, dx, dy, dz) = grid

    ix, iy, iz = @index(Global, NTuple)

    @inbounds begin
        # derivatives H:
        ix == 1 ? ixm1 = Nx : ixm1 = ix - 1
        iy == 1 ? iym1 = Ny : iym1 = iy - 1
        iz == 1 ? izm1 = Nz : izm1 = iz - 1
        dHxy = (Hx[ix,iy,iz] - Hx[ix,iym1,iz]) / dy
        dHxz = (Hx[ix,iy,iz] - Hx[ix,iy,izm1]) / dz
        dHyx = (Hy[ix,iy,iz] - Hy[ixm1,iy,iz]) / dx
        dHyz = (Hy[ix,iy,iz] - Hy[ix,iy,izm1]) / dz
        dHzx = (Hz[ix,iy,iz] - Hz[ixm1,iy,iz]) / dx
        dHzy = (Hz[ix,iy,iz] - Hz[ix,iym1,iz]) / dy

        # apply CPML:
        if ix <= xlayer1.ind   # x left layer [1:ix1]
            (; K, A, B) = xlayer1
            ixpml = ix
            psiHyx1[ixpml,iy,iz] = B[ixpml] * psiHyx1[ixpml,iy,iz] + A[ixpml] * dHyx
            psiHzx1[ixpml,iy,iz] = B[ixpml] * psiHzx1[ixpml,iy,iz] + A[ixpml] * dHzx
            Dy[ix,iy,iz] -= Md2[ix,iy,iz] * psiHzx1[ixpml,iy,iz]
            Dz[ix,iy,iz] += Md2[ix,iy,iz] * psiHyx1[ixpml,iy,iz]
            dHyx = dHyx / K[ixpml]
            dHzx = dHzx / K[ixpml]
        end
        if ix >= xlayer2.ind   # x right layer [ix2:Nx]
            (; ind, K, A, B) = xlayer2
            ixpml = ix - ind + 1
            psiHyx2[ixpml,iy,iz] = B[ixpml] * psiHyx2[ixpml,iy,iz] + A[ixpml] * dHyx
            psiHzx2[ixpml,iy,iz] = B[ixpml] * psiHzx2[ixpml,iy,iz] + A[ixpml] * dHzx
            Dy[ix,iy,iz] -= Md2[ix,iy,iz] * psiHzx2[ixpml,iy,iz]
            Dz[ix,iy,iz] += Md2[ix,iy,iz] * psiHyx2[ixpml,iy,iz]
            dHyx = dHyx / K[ixpml]
            dHzx = dHzx / K[ixpml]
        end
        if iy <= ylayer1.ind   # y left layer [1:iy1]
            (; K, A, B) = ylayer1
            iypml = iy
            psiHxy1[ix,iypml,iz] = B[iypml] * psiHxy1[ix,iypml,iz] + A[iypml] * dHxy
            psiHzy1[ix,iypml,iz] = B[iypml] * psiHzy1[ix,iypml,iz] + A[iypml] * dHzy
            Dx[ix,iy,iz] += Md2[ix,iy,iz] * psiHzy1[ix,iypml,iz]
            Dz[ix,iy,iz] -= Md2[ix,iy,iz] * psiHxy1[ix,iypml,iz]
            dHxy = dHxy / K[iypml]
            dHzy = dHzy / K[iypml]
        end
        if iy >= ylayer2.ind   # y right layer [iy2:Ny]
            (; ind, K, A, B) = ylayer2
            iypml = iy - ind + 1
            psiHxy2[ix,iypml,iz] = B[iypml] * psiHxy2[ix,iypml,iz] + A[iypml] * dHxy
            psiHzy2[ix,iypml,iz] = B[iypml] * psiHzy2[ix,iypml,iz] + A[iypml] * dHzy
            Dx[ix,iy,iz] += Md2[ix,iy,iz] * psiHzy2[ix,iypml,iz]
            Dz[ix,iy,iz] -= Md2[ix,iy,iz] * psiHxy2[ix,iypml,iz]
            dHxy = dHxy / K[iypml]
            dHzy = dHzy / K[iypml]
        end
        if iz <= zlayer1.ind   # z left layer [1:iz1]
            (; K, A, B) = zlayer1
            izpml = iz
            psiHxz1[ix,iy,izpml] = B[izpml] * psiHxz1[ix,iy,izpml] + A[izpml] * dHxz
            psiHyz1[ix,iy,izpml] = B[izpml] * psiHyz1[ix,iy,izpml] + A[izpml] * dHyz
            Dx[ix,iy,iz] -= Md2[ix,iy,iz] * psiHyz1[ix,iy,izpml]
            Dy[ix,iy,iz] += Md2[ix,iy,iz] * psiHxz1[ix,iy,izpml]
            dHxz = dHxz / K[izpml]
            dHyz = dHyz / K[izpml]
        end
        if iz >= zlayer2.ind   # z right layer [iz2:Nz]
            (; ind, K, A, B) = zlayer2
            izpml = iz - ind + 1
            psiHxz2[ix,iy,izpml] = B[izpml] * psiHxz2[ix,iy,izpml] + A[izpml] * dHxz
            psiHyz2[ix,iy,izpml] = B[izpml] * psiHyz2[ix,iy,izpml] + A[izpml] * dHyz
            Dx[ix,iy,iz] -= Md2[ix,iy,iz] * psiHyz2[ix,iy,izpml]
            Dy[ix,iy,iz] += Md2[ix,iy,iz] * psiHxz2[ix,iy,izpml]
            dHxz = dHxz / K[izpml]
            dHyz = dHyz / K[izpml]
        end

        # update D:
        Dx[ix,iy,iz] = Md1[ix,iy,iz] * Dx[ix,iy,iz] + Md2[ix,iy,iz] * (dHzy - dHyz)
        Dy[ix,iy,iz] = Md1[ix,iy,iz] * Dy[ix,iy,iz] + Md2[ix,iy,iz] * (dHxz - dHzx)
        Dz[ix,iy,iz] = Md1[ix,iy,iz] * Dz[ix,iy,iz] + Md2[ix,iy,iz] * (dHyx - dHxy)

        # update P:
        Nq = size(Px, 1)
        sumPx = zero(eltype(Px))
        sumPy = zero(eltype(Py))
        sumPz = zero(eltype(Pz))
        for iq=1:Nq
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
            sumPx += Px[iq,ix,iy,iz]
            sumPy += Py[iq,ix,iy,iz]
            sumPz += Pz[iq,ix,iy,iz]
        end

        # update E:
        Ex[ix,iy,iz] = Me[ix,iy,iz] * (Dx[ix,iy,iz] - sumPx)
        Ey[ix,iy,iz] = Me[ix,iy,iz] * (Dy[ix,iy,iz] - sumPy)
        Ez[ix,iy,iz] = Me[ix,iy,iz] * (Dz[ix,iy,iz] - sumPz)


        # update E explicit:
        # (; dt) = model
        # Ex[ix,iy,iz] += dt * Me[ix,iy,iz] * ((dHzy - dHyz) + (psiHzy[ix,iy,iz] - psiHyz[ix,iy,iz]))
        # Ey[ix,iy,iz] += dt * Me[ix,iy,iz] * ((dHxz - dHzx) + (psiHxz[ix,iy,iz] - psiHzx[ix,iy,iz]))
        # Ez[ix,iy,iz] += dt * Me[ix,iy,iz] * ((dHyx - dHxy) + (psiHyx[ix,iy,iz] - psiHxy[ix,iy,iz]))
    end
end
function update_E!(model::Model3D)
    (; Ex) = model.field
    backend = get_backend(Ex)
    ndrange = size(Ex)
    # In order to avoid the issue caused by the large size of the CUDA kernel parameters,
    # here we pass the parameters of the model explicitly:
    # https://discourse.julialang.org/t/passing-too-long-tuples-into-cuda-kernel-causes-an-error
    (; field, pml, Me, Md1, Md2,
       Aq, Bq, Cq, Px, oldPx1, oldPx2, Py, oldPy1, oldPy2, Pz, oldPz1, oldPz2) = model
    update_E_kernel!(backend)(
        field, pml, Me, Md1, Md2,
        Aq, Bq, Cq, Px, oldPx1, oldPx2, Py, oldPy1, oldPy2, Pz, oldPz1, oldPz2;
        ndrange
    )
    return nothing
end


# ******************************************************************************************
# Util
# ******************************************************************************************
"""
https://julialang.org/blog/2016/02/iteration/#a_multidimensional_boxcar_filter
"""
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
