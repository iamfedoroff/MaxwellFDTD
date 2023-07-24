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
    grid::Grid1D, source_data;
    tmax, CN=0.5, material=nothing, material_geometry=nothing, pml_box=(0,0),
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
        material_geometry = z -> false
        eps = 1
        mu = 1
        sigma = 0
        chi = [nothing]
    elseif isnothing(material_geometry)
        @error """

            You gave me the material, but did not tell where to place it.
            Please, specify the material geometry.
        """
    else
        (; eps, mu, sigma, chi) = material
    end

    # Permittivity, permeability, and conductivity:
    eps = [material_geometry(z[iz]) ? eps : 1 for iz=1:Nz]
    mu = [material_geometry(z[iz]) ? mu : 1 for iz=1:Nz]
    sigma = [material_geometry(z[iz]) ? sigma : 0 for iz=1:Nz]
    @. sigma = sigma / (EPS0*eps)   # J=sigma*E -> J=sigma*D

    # Variables for ADE dispersion calculation:
    Nq = length(chi)
    Aq, Bq, Cq = (zeros(Nq,Nz) for i=1:3)
    for iz=1:Nz, iq=1:Nq
        Aq0, Bq0, Cq0 = ade_coefficients(chi[iq], dt)
        Aq[iq,iz] = material_geometry(z[iz]) * Aq0
        Bq[iq,iz] = material_geometry(z[iz]) * Bq0
        Cq[iq,iz] = material_geometry(z[iz]) * Cq0
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
    (; iz1, iz2, Kz, Az, Bz, psiExz) = pml
    (; grid, Hy, Ex) = field
    (; Nz, dz) = grid

    iz = @index(Global)

    @inbounds begin
        # derivatives E:
        iz == Nz ? izp1 = 1 : izp1 = iz + 1
        dExz = (Ex[izp1] - Ex[iz]) / dz

        # apply CPML:
        if (iz <= iz1) || (iz >= iz2)
            psiExz[iz] = Bz[iz] * psiExz[iz] + Az[iz] * dExz
            Hy[iz] -= Mh[iz] * (0 + psiExz[iz])
        end

        # update H:
        Hy[iz] -= Mh[iz] * (0 + dExz / Kz[iz])
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
    (; iz1, iz2, Kz, Az, Bz, psiHyz) = pml
    (; grid, Hy, Dx, Ex) = field
    (; Nz, dz) = grid

    iz = @index(Global)

    @inbounds begin
        # derivatives H:
        iz == 1 ? izm1 = Nz : izm1 = iz - 1
        dHyz = (Hy[iz] - Hy[izm1]) / dz

        # apply CPML:
        if (iz <= iz1) || (iz >= iz2)
            psiHyz[iz] = Bz[iz] * psiHyz[iz] + Az[iz] * dHyz
            Dx[iz] += Md2[iz] * (0 - psiHyz[iz])
        end

        # update D:
        Dx[iz] = Md1[iz] * Dx[iz] + Md2[iz] * (0 - dHyz / Kz[iz])

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
    grid::Grid2D, source_data;
    tmax, CN=0.5, material=nothing, material_geometry=nothing, pml_box=(0,0,0,0),
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
        material_geometry = (x,z) -> false
        eps = 1
        mu = 1
        sigma = 0
        chi = [nothing]
    elseif isnothing(material_geometry)
        @error """

            You gave me the material, but did not tell where to place it.
            Please, specify the material geometry.
        """
    else
        (; eps, mu, sigma, chi) = material
    end

    # Permittivity, permeability, and conductivity:
    eps = [material_geometry(x[ix],z[iz]) ? eps : 1 for ix=1:Nx, iz=1:Nz]
    mu = [material_geometry(x[ix],z[iz]) ? mu : 1 for ix=1:Nx, iz=1:Nz]
    sigma = [material_geometry(x[ix],z[iz]) ? sigma : 0 for ix=1:Nx, iz=1:Nz]
    @. sigma = sigma / (EPS0*eps)   # J=sigma*E -> J=sigma*D

    # Variables for ADE dispersion calculation:
    Nq = length(chi)
    Aq, Bq, Cq = (zeros(Nq,Nx,Nz) for i=1:3)
    for iz=1:Nz, ix=1:Nx, iq=1:Nq
        Aq0, Bq0, Cq0 = ade_coefficients(chi[iq], dt)
        Aq[iq,ix,iz] = material_geometry(x[ix],z[iz]) * Aq0
        Bq[iq,ix,iz] = material_geometry(x[ix],z[iz]) * Bq0
        Cq[iq,ix,iz] = material_geometry(x[ix],z[iz]) * Cq0
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
    (; ix1, ix2, iz1, iz2, Kx, Ax, Bx, Kz, Az, Bz, psiExz, psiEzx) = pml
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
        if (ix <= ix1) || (ix >= ix2)
            psiEzx[ix,iz] = Bx[ix] * psiEzx[ix,iz] + Ax[ix] * dEzx
            Hy[ix,iz] += Mh[ix,iz] * psiEzx[ix,iz]
        end
        if (iz <= iz1) || (iz >= iz2)
            psiExz[ix,iz] = Bz[iz] * psiExz[ix,iz] + Az[iz] * dExz
            Hy[ix,iz] -= Mh[ix,iz] * psiExz[ix,iz]

        end

        # update H:
        Hy[ix,iz] -= Mh[ix,iz] * (dExz / Kz[iz] - dEzx / Kx[ix])
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
    (; ix1, ix2, iz1, iz2, Kx, Ax, Bx, Kz, Az, Bz, psiHyx, psiHyz) = pml
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
        if (ix <= ix1) || (ix >= ix2)
            psiHyx[ix,iz] = Bx[ix] * psiHyx[ix,iz] + Ax[ix] * dHyx
            Dz[ix,iz] += Md2[ix,iz] * psiHyx[ix,iz]
        end
        if (iz <= iz1) || (iz >= iz2)
            psiHyz[ix,iz] = Bz[iz] * psiHyz[ix,iz] + Az[iz] * dHyz
            Dx[ix,iz] -= Md2[ix,iz] * psiHyz[ix,iz]
        end

        # update D:
        Dx[ix,iz] = Md1[ix,iz] * Dx[ix,iz] + Md2[ix,iz] * (0 - dHyz / Kz[iz])
        Dz[ix,iz] = Md1[ix,iz] * Dz[ix,iz] + Md2[ix,iz] * (dHyx / Kx[ix] - 0)

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
    grid::Grid3D, source_data;
    tmax, CN=0.5, material=nothing, material_geometry=nothing, pml_box=(0,0,0,0,0,0),
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
        material_geometry = (x,y,z) -> false
        eps = 1
        mu = 1
        sigma = 0
        chi = [nothing]
    elseif isnothing(material_geometry)
        @error """

            You gave me the material, but did not tell where to place it.
            Please, specify the material geometry.
        """
    else
        (; eps, mu, sigma, chi) = material
    end

    # Permittivity, permeability, and conductivity:
    eps = [material_geometry(x[ix],y[iy],z[iz]) ? eps : 1 for ix=1:Nx, iy=1:Ny, iz=1:Nz]
    mu = [material_geometry(x[ix],y[iy],z[iz]) ? mu : 1 for ix=1:Nx, iy=1:Ny, iz=1:Nz]
    sigma = [material_geometry(x[ix],y[iy],z[iz]) ? sigma : 0 for ix=1:Nx, iy=1:Ny, iz=1:Nz]
    @. sigma = sigma / (EPS0*eps)   # J=sigma*E -> J=sigma*D

    # Variables for ADE dispersion calculation:
    Nq = length(chi)
    Aq, Bq, Cq = (zeros(Nq,Nx,Ny,Nz) for i=1:3)
    for iz=1:Nz, iy=1:Ny, ix=1:Nx, iq=1:Nq
        Aq0, Bq0, Cq0 = ade_coefficients(chi[iq], dt)
        Aq[iq,ix,iy,iz] = material_geometry(x[ix],y[iy],z[iz]) * Aq0
        Bq[iq,ix,iy,iz] = material_geometry(x[ix],y[iy],z[iz]) * Bq0
        Cq[iq,ix,iy,iz] = material_geometry(x[ix],y[iy],z[iz]) * Cq0
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


@kernel function update_H_kernel!(model::Model3D)
    (; field, pml, Mh) = model
    (; ix1, ix2, iy1, iy2, iz1, iz2, Kx, Ax, Bx, Ky, Ay, By, Kz, Az, Bz) = pml
    (; psiExy, psiExz, psiEyx, psiEyz, psiEzx, psiEzy) = pml
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
        if (ix <= ix1) || (ix >= ix2)
            psiEyx[ix,iy,iz] = Bx[ix] * psiEyx[ix,iy,iz] + Ax[ix] * dEyx
            psiEzx[ix,iy,iz] = Bx[ix] * psiEzx[ix,iy,iz] + Ax[ix] * dEzx
            Hy[ix,iy,iz] += Mh[ix,iy,iz] * psiEzx[ix,iy,iz]
            Hz[ix,iy,iz] -= Mh[ix,iy,iz] * psiEyx[ix,iy,iz]
        end
        if (iy <= iy1) || (iy >= iy2)
            psiExy[ix,iy,iz] = By[iy] * psiExy[ix,iy,iz] + Ay[iy] * dExy
            psiEzy[ix,iy,iz] = By[iy] * psiEzy[ix,iy,iz] + Ay[iy] * dEzy
            Hx[ix,iy,iz] -= Mh[ix,iy,iz] * psiEzy[ix,iy,iz]
            Hz[ix,iy,iz] += Mh[ix,iy,iz] * psiExy[ix,iy,iz]
        end
        if (iz <= iz1) || (iz >= iz2)
            psiExz[ix,iy,iz] = Bz[iz] * psiExz[ix,iy,iz] + Az[iz] * dExz
            psiEyz[ix,iy,iz] = Bz[iz] * psiEyz[ix,iy,iz] + Az[iz] * dEyz
            Hx[ix,iy,iz] += Mh[ix,iy,iz] * psiEyz[ix,iy,iz]
            Hy[ix,iy,iz] -= Mh[ix,iy,iz] * psiExz[ix,iy,iz]
        end

        # update H:
        Hx[ix,iy,iz] -= Mh[ix,iy,iz] * (dEzy / Ky[iy] - dEyz / Kz[iz])
        Hy[ix,iy,iz] -= Mh[ix,iy,iz] * (dExz / Kz[iz] - dEzx / Kx[ix])
        Hz[ix,iy,iz] -= Mh[ix,iy,iz] * (dEyx / Kx[ix] - dExy / Ky[iy])
    end
end
function update_H!(model::Model3D)
    (; Hx) = model.field
    backend = get_backend(Hx)
    ndrange = size(Hx)
    update_H_kernel!(backend)(model; ndrange)
    return nothing
end


@kernel function update_E_kernel!(model::Model3D)
    (; field, pml, Me, Md1, Md2) = model
    (; Aq, Bq, Cq, Px, oldPx1, oldPx2, Py, oldPy1, oldPy2, Pz, oldPz1, oldPz2) = model
    (; ix1, ix2, iy1, iy2, iz1, iz2, Kx, Ax, Bx, Ky, Ay, By, Kz, Az, Bz) = pml
    (; psiHxy, psiHxz, psiHyx, psiHyz, psiHzx, psiHzy) = pml
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
        if (ix <= ix1) || (ix >= ix2)
            psiHyx[ix,iy,iz] = Bx[ix] * psiHyx[ix,iy,iz] + Ax[ix] * dHyx
            psiHzx[ix,iy,iz] = Bx[ix] * psiHzx[ix,iy,iz] + Ax[ix] * dHzx
            Dy[ix,iy,iz] -= Md2[ix,iy,iz] * psiHzx[ix,iy,iz]
            Dz[ix,iy,iz] += Md2[ix,iy,iz] * psiHyx[ix,iy,iz]
        end
        if (iy <= iy1) || (iy >= iy2)
            psiHxy[ix,iy,iz] = By[iy] * psiHxy[ix,iy,iz] + Ay[iy] * dHxy
            psiHzy[ix,iy,iz] = By[iy] * psiHzy[ix,iy,iz] + Ay[iy] * dHzy
            Dx[ix,iy,iz] += Md2[ix,iy,iz] * psiHzy[ix,iy,iz]
            Dz[ix,iy,iz] -= Md2[ix,iy,iz] * psiHxy[ix,iy,iz]
        end
        if (iz <= iz1) || (iz >= iz2)
            psiHxz[ix,iy,iz] = Bz[iz] * psiHxz[ix,iy,iz] + Az[iz] * dHxz
            psiHyz[ix,iy,iz] = Bz[iz] * psiHyz[ix,iy,iz] + Az[iz] * dHyz
            Dx[ix,iy,iz] -= Md2[ix,iy,iz] * psiHyz[ix,iy,iz]
            Dy[ix,iy,iz] += Md2[ix,iy,iz] * psiHxz[ix,iy,iz]
        end

        # update D:
        Dx[ix,iy,iz] = Md1[ix,iy,iz] * Dx[ix,iy,iz] +
                       Md2[ix,iy,iz] * (dHzy / Ky[iy] - dHyz / Kz[iz])
        Dy[ix,iy,iz] = Md1[ix,iy,iz] * Dy[ix,iy,iz] +
                       Md2[ix,iy,iz] * (dHxz / Kz[iz] - dHzx / Kx[ix])
        Dz[ix,iy,iz] = Md1[ix,iy,iz] * Dz[ix,iy,iz] +
                       Md2[ix,iy,iz] * (dHyx / Kx[ix] - dHxy / Ky[iy])

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
    update_E_kernel!(backend)(model; ndrange)
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
