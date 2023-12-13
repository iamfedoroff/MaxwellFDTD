struct Model{F, S, P, M, T, R}
    field :: F
    sources :: S
    pml :: P
    material :: M
    # Time grid:
    Nt :: Int
    dt :: T
    t :: R
    # Update coefficients for H, E and D fields in vacuum:
    Mh :: T
    Me :: T
    Md1 :: T
    Md2 :: T
end

@adapt_structure Model


"""
Model(grid, source; tmax, CN=0.5, pml_box=nothing, material=nothing)

The model contains all data necessary to run FDTD simulaton.

# Arguments
- `grid::Grid`: Spatial grid.
- `source::Source`: Light source.

# Keywords
- `tmax::Real`: Duration of FDTD simulation in seconds.
- `CN::Real=0.5`: Courant number which defines the size of the temporal step
- `pml_box::Tuple=nothing`: The thicknesses of the PML layers along each grid coordinate,
    e.g. in 1D pml_box=(1e-2,2e-2) will define 1 cm thick PML layer at the left and 2 cm
    thick PML layer at the right edge of z coordinate. If not provided, then periodic
    boundary conditions are used.
- `material::Material=nothing`: Structure with material properties. If not provided, then
    FDTD simulation is performed in free space.
"""
function Model(grid, source; tmax, CN=0.5, pml_box=nothing, material=nothing)
    field = Field(grid)

    # Time grid:
    dt = time_step(grid, CN)
    t = range(start=0, step=dt, stop=tmax)
    Nt = length(t)

    if source isa Source
        source = (source,)
    end
    sources = Tuple([SourceStruct(s, field, t) for s in source])

    pml = PML(grid, pml_box, dt)

    material = MaterialStruct(material, grid, dt)

    # Compensation for the numerical dispersion:
    # dt = t[2] - t[1]
    # lam0 = 2e-6   # (m) wavelength
    # w0 = 2*pi * C0 / lam0   # frequency
    # sn = C0 * dt/dz * sin(w0/C0 * dz/2) / sin(w0 * dt/2)
    # @show sn
    # eps = @. sn * eps
    # mu = @. sn * mu

    # Update coefficients for H, E and D fields in vacuum:
    Mh = dt / MU0
    Me = 1 / EPS0
    Md1 = 1.0
    Md2 = dt

    return Model(field, sources, pml, material, Nt, dt, t, Mh, Me, Md1, Md2)
end


function step!(model, it)
    update_H!(model)
    update_E!(model)
    for source in model.sources
        add_source!(model, source, it)
    end
    return nothing
end


"""
    solve!(model; kwargs...) -> Model

Runs the FDTD simulation and returns the updated model structure. The keyword arguments
specify the outputs.

# Arguments
- `model::Model`: Model structure with parameters of FDTD simulation.

# Keywords
- `arch::ARCH=CPU()`: Architecture (CPU or GPU) to run the simulations on.
- `fname::String="out.hdf"`: The name of the output file.
- `nframes::Int=nothing`: The total number of time frames at which the components of the
    field will be written to the output file.
- `nstride::Int=nothing`: The components of the field will be written to the output file
    every 'nstride' time steps.
- `dtout::Real=nothing`: The components of the field will be written to the output file
    every 'dtout' time step.
- `tfsf_record::Bool=false`: If true, then write TFSF source data into a file.
- `tfsf_fname::String=nothing`: The name of the file where to write the TFSF data.
- `tfsf_box::Tuple=nothing`: The coordinates of the TFSF box faces.
- `viewpoints::Tuple{Tuple}=nothing`: List of viewpoints.

# Returns
- `Model`: Updated model.
"""
function solve!(
    model;
    arch=CPU(),
    fname="out.hdf",
    nframes=nothing,
    nstride=nothing,
    dtout=nothing,
    tfsf_record=false,
    tfsf_fname=nothing,
    tfsf_box=nothing,
    viewpoints=nothing,
)
    model = adapt(arch, model)
    (; Nt, dt, t) = model

    out = Output(model; fname, nstride, nframes, dtout, viewpoints)

    if tfsf_record
        if isnothing(tfsf_fname)
            ext = splitext(out.fname)[end]
            tfsf_fname = replace(out.fname, ext => "_tfsf" * ext)
        end
        if !isdir(dirname(tfsf_fname))
            mkpath(dirname(tfsf_fname))
        end
        tfsf_data = prepare_tfsf_record(model, tfsf_fname, tfsf_box)
    end

    @showprogress 1 for it=1:Nt
        @timeit "model step" begin
            step!(model, it)
            if CUDA.functional()
                synchronize()
            end
        end

        @timeit "output" begin
            if (out.itout <= out.Ntout) && (abs(t[it] - out.tout[out.itout]) <= dt/2)
                write_fields(out, model)
                out.itout += 1
            end
            write_viewpoints(out, model, it)
            calculate_output_variables!(out, model)
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

    write_output_variables(out, model)

    print_timer()

    return model
end


# ******************************************************************************************
# 1D: d/dx = d/dy = 0,   (Hy, Ex)
# ******************************************************************************************
@kernel function update_H_kernel!(model::Model{F}) where F <: Field1D
    (; field, pml, material) = model
    (; grid, Hy, Ex) = field
    (; Nz, dz) = grid
    (; zlayer1, psiExz1, zlayer2, psiExz2) = pml
    (; geometry) = material

    iz = @index(Global)

    @inbounds begin
        isgeometry = geometry[iz]

        if isgeometry
            (; Mh) = material   # Mh=dt/(MU0*mu)
        else
            (; Mh) = model   # Mh=dt/MU0
        end

        # derivatives E ....................................................................
        iz == Nz ? izp1 = 1 : izp1 = iz + 1
        dExz = (Ex[izp1] - Ex[iz]) / dz

        # apply CPML .......................................................................
        if iz <= zlayer1.ind   # z left layer [1:iz1]
            (; K, A, B) = zlayer1
            izpml = iz
            psiExz1[izpml] = B[izpml] * psiExz1[izpml] + A[izpml] * dExz
            Hy[iz] -= Mh * psiExz1[izpml]
            dExz = dExz / K[izpml]
        end
        if iz >= zlayer2.ind   # z right layer [iz1:Nz]
            (; ind, K, A, B) = zlayer2
            izpml = iz - ind + 1
            psiExz2[izpml] = B[izpml] * psiExz2[izpml] + A[izpml] * dExz
            Hy[iz] -= Mh * psiExz2[izpml]
            dExz = dExz / K[izpml]
        end

        # update H .........................................................................
        Hy[iz] -= Mh * (0 + dExz)
    end
end
function update_H!(model::Model{F}) where F <: Field1D
    (; Hy) = model.field
    backend = get_backend(Hy)
    ndrange = size(Hy)
    update_H_kernel!(backend)(model; ndrange)
    return nothing
end


@kernel function update_E_kernel!(model::Model{F}) where F <: Field1D
    (; field, pml, material, dt) = model
    (; grid, Hy, Dx, Ex) = field
    (; Nz, dz) = grid
    (; zlayer1, psiHyz1, zlayer2, psiHyz2) = pml
    (; geometry, isdispersion, iskerr, isplasma) = material

    iz = @index(Global)

    @inbounds begin
        isgeometry = geometry[iz]

        if isgeometry
            # Me=1/(EPS0*eps), Md1=(1-sigma*dt/2)/(1+sigma*dt/2), Md2=dt/(1+sigma*dt/2)
            (; Me, Md1, Md2) = material
        else
            # Me=1/EPS0, Md1=1, Md2=dt
            (; Me, Md1, Md2) = model
        end

        # derivatives H ....................................................................
        iz == 1 ? izm1 = Nz : izm1 = iz - 1
        dHyz = (Hy[iz] - Hy[izm1]) / dz

        # apply CPML .......................................................................
        if iz <= zlayer1.ind   # z left layer [1:iz1]
            (; K, A, B) = zlayer1
            izpml = iz
            psiHyz1[izpml] = B[izpml] * psiHyz1[izpml] + A[izpml] * dHyz
            Dx[iz] -= Md2 * psiHyz1[izpml]
            dHyz = dHyz / K[izpml]
        end
        if iz >= zlayer2.ind   # z right layer [iz1:Nz]
            (; ind, K, A, B) = zlayer2
            izpml = iz - ind + 1
            psiHyz2[izpml] = B[izpml] * psiHyz2[izpml] + A[izpml] * dHyz
            Dx[iz] -= Md2 * psiHyz2[izpml]
            dHyz = dHyz / K[izpml]
        end

        # update D .........................................................................
        Dx[iz] = Md1 * Dx[iz] + Md2 * (0 - dHyz)

        # materials ........................................................................
        sumPx = zero(eltype(Ex))

        # linear polarization:
        if isdispersion && isgeometry
            (; Aq, Bq, Cq, Px, oldPx1, oldPx2) = material
            Nq = size(Px, 1)
            for iq=1:Nq
                oldPx2[iq,iz] = oldPx1[iq,iz]
                oldPx1[iq,iz] = Px[iq,iz]
                Px[iq,iz] = Aq[iq] * Px[iq,iz] + Bq[iq] * oldPx2[iq,iz] + Cq[iq] * Ex[iz]
                sumPx += Px[iq,iz]
            end
        end

        # plasma:
        if isplasma && isgeometry
            (; ionrate, Rava, ksi, rho0, rho, drho,
               Ap, Bp, Cp, Ppx, oldPpx1, oldPpx2, Ma, Pax) = material

            E2 = abs2(Ex[iz])
            II = ksi * E2   # intensity

            # plasma current:
            oldPpx2[iz] = oldPpx1[iz]
            oldPpx1[iz] = Ppx[iz]
            Ppx[iz] = Ap * Ppx[iz] + Bp * oldPpx2[iz] + Cp * rho[iz]*rho0 * Ex[iz]
            sumPx += Ppx[iz]

            # multi-photon ionization losses:
            if E2 >= eps(one(E2))
                invE2 = 1 / E2
            else
                invE2 = zero(E2)
            end
            Pax[iz] += Ma * drho[iz]*rho0 * Ex[iz] * invE2
            sumPx += Pax[iz]

            # electron density:
            R1 = ionrate(II)
            R2 = Rava * E2
            if R2 == 0
                rho[iz] = 1 - (1 - rho[iz]) * exp(-R1 * dt)
            else
                R12 = R1 - R2
                rho[iz] = R1/R12*1 - (R1/R12*1 - rho[iz]) * exp(-R12 * dt)
            end
            drho[iz] = R1 * (1 - rho[iz])
        end

        # update E .........................................................................
        DmPx = Dx[iz] - sumPx

        if iskerr && isgeometry
            (; Mk2, Mk3) = material   # Mk2=EPS0*chi2, Mk3=EPS0*chi3

            # Kerr by Meep [A.F. Oskooi, Comput. Phys. Commun., 181, 687 (2010)]
            Ex[iz] = (1 + 1*Mk2 * Me^2 * DmPx + 2*Mk3 * Me^3 * DmPx^2) /
                     (1 + 2*Mk2 * Me^2 * DmPx + 3*Mk3 * Me^3 * DmPx^2) * DmPx * Me
        else
            Ex[iz] = DmPx * Me
        end

        # update E explicit:
        # (; dt) = model
        # Ex[iz] += dt * Me * ((0 - dHyz / Kz[iz]) + (0 - psiHyz[iz]))
    end
end
function update_E!(model::Model{F}) where F <: Field1D
    (; Ex) = model.field
    backend = get_backend(Ex)
    ndrange = size(Ex)
    update_E_kernel!(backend)(model; ndrange)
    return nothing
end


# ******************************************************************************************
# 2D
# ******************************************************************************************
@kernel function update_H_kernel!(model::Model{F}) where F <: Field2D
    (; field, pml, material) = model
    (; grid, Hy, Ex, Ez) = field
    (; Nx, Nz, dx, dz) = grid
    (; xlayer1, psiEzx1, xlayer2, psiEzx2, zlayer1, psiExz1, zlayer2, psiExz2) = pml
    (; geometry) = material

    ix, iz = @index(Global, NTuple)

    @inbounds begin
        isgeometry = geometry[ix,iz]

        if isgeometry
            (; Mh) = material   # Mh=dt/(MU0*mu)
        else
            (; Mh) = model   # Mh=dt/MU0
        end

        # derivatives E ....................................................................
        ix == Nx ? ixp1 = 1 : ixp1 = ix + 1
        iz == Nz ? izp1 = 1 : izp1 = iz + 1
        dExz = (Ex[ix,izp1] - Ex[ix,iz]) / dz
        dEzx = (Ez[ixp1,iz] - Ez[ix,iz]) / dx

        # apply CPML .......................................................................
        if ix <= xlayer1.ind   # x left layer [1:ix1]
            (; K, A, B) = xlayer1
            ixpml = ix
            psiEzx1[ixpml,iz] = B[ixpml] * psiEzx1[ixpml,iz] + A[ixpml] * dEzx
            Hy[ix,iz] += Mh * psiEzx1[ixpml,iz]
            dEzx = dEzx / K[ixpml]
        end
        if ix >= xlayer2.ind      # x right layer [ix2:Nx]
            (; ind, K, A, B) = xlayer2
            ixpml = ix - ind + 1
            psiEzx2[ixpml,iz] = B[ixpml] * psiEzx2[ixpml,iz] + A[ixpml] * dEzx
            Hy[ix,iz] += Mh * psiEzx2[ixpml,iz]
            dEzx = dEzx / K[ixpml]
        end
        if iz <= zlayer1.ind   # z left layer [1:iz1]
            (; K, A, B) = zlayer1
            izpml = iz
            psiExz1[ix,izpml] = B[izpml] * psiExz1[ix,izpml] + A[izpml] * dExz
            Hy[ix,iz] -= Mh * psiExz1[ix,izpml]
            dExz = dExz / K[izpml]
        end
        if iz >= zlayer2.ind      # z right layer [iz1:Nz]
            (; ind, K, A, B) = zlayer2
            izpml = iz - ind + 1
            psiExz2[ix,izpml] = B[izpml] * psiExz2[ix,izpml] + A[izpml] * dExz
            Hy[ix,iz] -= Mh * psiExz2[ix,izpml]
            dExz = dExz / K[izpml]
        end

        # update H .........................................................................
        Hy[ix,iz] -= Mh * (dExz - dEzx)
    end
end
function update_H!(model::Model{F}) where F <: Field2D
    (; Hy) = model.field
    backend = get_backend(Hy)
    ndrange = size(Hy)
    update_H_kernel!(backend)(model; ndrange)
    return nothing
end


@kernel function update_E_kernel!(model::Model{F}) where F <: Field2D
    (; field, pml, material, dt) = model
    (; grid, Hy, Dx, Dz, Ex, Ez) = field
    (; Nx, Nz, dx, dz) = grid
    (; xlayer1, psiHyx1, xlayer2, psiHyx2, zlayer1, psiHyz1, zlayer2, psiHyz2) = pml
    (; geometry, isdispersion, iskerr, isplasma) = material

    ix, iz = @index(Global, NTuple)

    @inbounds begin
        isgeometry = geometry[ix,iz]

        if isgeometry
            # Me=1/(EPS0*eps), Md1=(1-sigma*dt/2)/(1+sigma*dt/2), Md2=dt/(1+sigma*dt/2)
            (; Me, Md1, Md2) = material
        else
            # Me=1/EPS0, Md1=1, Md2=dt
            (; Me, Md1, Md2) = model
        end

        # derivatives H ....................................................................
        ix == 1 ? ixm1 = Nx : ixm1 = ix - 1
        iz == 1 ? izm1 = Nz : izm1 = iz - 1
        dHyx = (Hy[ix,iz] - Hy[ixm1,iz]) / dx
        dHyz = (Hy[ix,iz] - Hy[ix,izm1]) / dz

        # apply CPML .......................................................................
        if ix <= xlayer1.ind   # x left layer [1:ix1]
            (; K, A, B) = xlayer1
            ixpml = ix
            psiHyx1[ixpml,iz] = B[ixpml] * psiHyx1[ixpml,iz] + A[ixpml] * dHyx
            Dz[ix,iz] += Md2 * psiHyx1[ixpml,iz]
            dHyx = dHyx / K[ixpml]
        end
        if ix >= xlayer2.ind   # x right layer [ix2:Nx]
            (; ind, K, A, B) = xlayer2
            ixpml = ix - ind + 1
            psiHyx2[ixpml,iz] = B[ixpml] * psiHyx2[ixpml,iz] + A[ixpml] * dHyx
            Dz[ix,iz] += Md2 * psiHyx2[ixpml,iz]
            dHyx = dHyx / K[ixpml]
        end
        if iz <= zlayer1.ind   # z left layer [1:iz1]
            (; K, A, B) = zlayer1
            izpml = iz
            psiHyz1[ix,izpml] = B[izpml] * psiHyz1[ix,izpml] + A[izpml] * dHyz
            Dx[ix,iz] -= Md2 * psiHyz1[ix,izpml]
            dHyz = dHyz / K[izpml]
        end
        if iz >= zlayer2.ind   # z right layer [iz2:Nz]
            (; ind, K, A, B) = zlayer2
            izpml = iz - ind + 1
            psiHyz2[ix,izpml] = B[izpml] * psiHyz2[ix,izpml] + A[izpml] * dHyz
            Dx[ix,iz] -= Md2 * psiHyz2[ix,izpml]
            dHyz = dHyz / K[izpml]
        end

        # update D .........................................................................
        Dx[ix,iz] = Md1 * Dx[ix,iz] + Md2 * (0 - dHyz)
        Dz[ix,iz] = Md1 * Dz[ix,iz] + Md2 * (dHyx - 0)

        # materials ........................................................................
        sumPx = zero(eltype(Ex))
        sumPz = zero(eltype(Ez))

        # linear polarization:
        if isdispersion && isgeometry
            (; Aq, Bq, Cq, Px, oldPx1, oldPx2, Pz, oldPz1, oldPz2) = material
            Nq = size(Px, 1)
            for iq=1:Nq
                oldPx2[iq,ix,iz] = oldPx1[iq,ix,iz]
                oldPx1[iq,ix,iz] = Px[iq,ix,iz]
                Px[iq,ix,iz] = Aq[iq] * Px[iq,ix,iz] +
                               Bq[iq] * oldPx2[iq,ix,iz] +
                               Cq[iq] * Ex[ix,iz]
                oldPz2[iq,ix,iz] = oldPz1[iq,ix,iz]
                oldPz1[iq,ix,iz] = Pz[iq,ix,iz]
                Pz[iq,ix,iz] = Aq[iq] * Pz[iq,ix,iz] +
                               Bq[iq] * oldPz2[iq,ix,iz] +
                               Cq[iq] * Ez[ix,iz]
                sumPx += Px[iq,ix,iz]
                sumPz += Pz[iq,ix,iz]
            end
        end

        # plasma:
        if isplasma && isgeometry
            (; ionrate, Rava, ksi, rho0, rho, drho, Ap, Bp, Cp,
               Ppx, oldPpx1, oldPpx2, Ppz, oldPpz1, oldPpz2, Ma, Pax, Paz) = material

            E2 = abs2(Ex[ix,iz]) + abs2(Ez[ix,iz])
            II = ksi * E2   # intensity

            # plasma current:
            oldPpx2[ix,iz] = oldPpx1[ix,iz]
            oldPpx1[ix,iz] = Ppx[ix,iz]
            Ppx[ix,iz] = Ap * Ppx[ix,iz] +
                         Bp * oldPpx2[ix,iz] +
                         Cp * rho[ix,iz]*rho0 * Ex[ix,iz]
            oldPpz2[ix,iz] = oldPpz1[ix,iz]
            oldPpz1[ix,iz] = Ppz[ix,iz]
            Ppz[ix,iz] = Ap * Ppz[ix,iz] +
                         Bp * oldPpz2[ix,iz] +
                         Cp * rho[ix,iz]*rho0 * Ez[ix,iz]
            sumPx += Ppx[ix,iz]
            sumPz += Ppz[ix,iz]

            # multi-photon ionization losses:
            if E2 >= eps(one(E2))
                invE2 = 1 / E2
            else
                invE2 = zero(E2)
            end
            Pax[ix,iz] += Ma * drho[ix,iz]*rho0 * Ex[ix,iz] * invE2
            Paz[ix,iz] += Ma * drho[ix,iz]*rho0 * Ez[ix,iz] * invE2
            sumPx += Pax[ix,iz]
            sumPz += Paz[ix,iz]

            # electron density:
            R1 = ionrate(II)
            R2 = Rava * E2
            if R2 == 0
                rho[ix,iz] = 1 - (1 - rho[ix,iz]) * exp(-R1 * dt)
            else
                R12 = R1 - R2
                rho[ix,iz] = R1/R12*1 - (R1/R12*1 - rho[ix,iz]) * exp(-R12 * dt)
            end
            drho[ix,iz] = R1 * (1 - rho[ix,iz])
        end

        # update E .........................................................................
        DmPx = Dx[ix,iz] - sumPx
        DmPz = Dz[ix,iz] - sumPz

        if iskerr && isgeometry
            (; Mk2, Mk3) = material   # Mk2=EPS0*chi2, Mk3=EPS0*chi3

            # Kerr by Meep [A.F. Oskooi, Comput. Phys. Commun., 181, 687 (2010)]
            Ex[ix,iz] = (1 + 1*Mk2 * Me^2 * DmPx + 2*Mk3 * Me^3 * DmPx^2) /
                        (1 + 2*Mk2 * Me^2 * DmPx + 3*Mk3 * Me^3 * DmPx^2) * DmPx * Me
            Ez[ix,iz] = (1 + 1*Mk2 * Me^2 * DmPz + 2*Mk3 * Me^3 * DmPz^2) /
                        (1 + 2*Mk2 * Me^2 * DmPz + 3*Mk3 * Me^3 * DmPz^2) * DmPz * Me
        else
            Ex[ix,iz] = DmPx * Me
            Ez[ix,iz] = DmPz * Me
        end

        # update E explicit:
        # (; dt) = model
        # Ex[ix,iz] += dt * Me * ((0 - dHyz / Kz[iz]) + (0 - psiHyz[ix,iz]))
        # Ez[ix,iz] += dt * Me * ((dHyx / Kx[ix] - 0) + (psiHyx[ix,iz] - 0))
    end
end
function update_E!(model::Model{F}) where F <: Field2D
    (; Ex) = model.field
    backend = get_backend(Ex)
    ndrange = size(Ex)
    update_E_kernel!(backend)(model; ndrange)
    return nothing
end


# ******************************************************************************************
# 3D
# ******************************************************************************************
# In order to avoid the issue caused by the large size of the CUDA kernel parameters,
# here we pass the parameters of the model explicitly:
# https://discourse.julialang.org/t/passing-too-long-tuples-into-cuda-kernel-causes-an-error
@kernel function update_H_kernel!(field::Field3D, pml, material, Mh0)
    (; grid, Hx, Hy, Hz, Ex, Ey, Ez) = field
    (; Nx, Ny, Nz, dx, dy, dz) = grid
    (; xlayer1, psiEyx1, psiEzx1, xlayer2, psiEyx2, psiEzx2,
       ylayer1, psiExy1, psiEzy1, ylayer2, psiExy2, psiEzy2,
       zlayer1, psiExz1, psiEyz1, zlayer2, psiExz2, psiEyz2) = pml
    (; geometry) = material

    ix, iy, iz = @index(Global, NTuple)

    @inbounds begin
        isgeometry = geometry[ix,iy,iz]

        if isgeometry
            (; Mh) = material   # Mh=dt/(MU0*mu)
        else
            Mh = Mh0   # Mh=dt/MU0
        end

        # derivatives E ....................................................................
        ix == Nx ? ixp1 = 1 : ixp1 = ix + 1
        iy == Ny ? iyp1 = 1 : iyp1 = iy + 1
        iz == Nz ? izp1 = 1 : izp1 = iz + 1
        dExy = (Ex[ix,iyp1,iz] - Ex[ix,iy,iz]) / dy
        dExz = (Ex[ix,iy,izp1] - Ex[ix,iy,iz]) / dz
        dEyx = (Ey[ixp1,iy,iz] - Ey[ix,iy,iz]) / dx
        dEyz = (Ey[ix,iy,izp1] - Ey[ix,iy,iz]) / dz
        dEzx = (Ez[ixp1,iy,iz] - Ez[ix,iy,iz]) / dx
        dEzy = (Ez[ix,iyp1,iz] - Ez[ix,iy,iz]) / dy

        # apply CPML .......................................................................
        if ix <= xlayer1.ind   # x left layer [1:ix1]
            (; K, A, B) = xlayer1
            ixpml = ix
            psiEyx1[ixpml,iy,iz] = B[ixpml] * psiEyx1[ixpml,iy,iz] + A[ixpml] * dEyx
            psiEzx1[ixpml,iy,iz] = B[ixpml] * psiEzx1[ixpml,iy,iz] + A[ixpml] * dEzx
            Hy[ix,iy,iz] += Mh * psiEzx1[ixpml,iy,iz]
            Hz[ix,iy,iz] -= Mh * psiEyx1[ixpml,iy,iz]
            dEyx = dEyx / K[ixpml]
            dEzx = dEzx / K[ixpml]
        end
        if ix >= xlayer2.ind   # x right layer [ix2:Nx]
            (; ind, K, A, B) = xlayer2
            ixpml = ix - ind + 1
            psiEyx2[ixpml,iy,iz] = B[ixpml] * psiEyx2[ixpml,iy,iz] + A[ixpml] * dEyx
            psiEzx2[ixpml,iy,iz] = B[ixpml] * psiEzx2[ixpml,iy,iz] + A[ixpml] * dEzx
            Hy[ix,iy,iz] += Mh * psiEzx2[ixpml,iy,iz]
            Hz[ix,iy,iz] -= Mh * psiEyx2[ixpml,iy,iz]
            dEyx = dEyx / K[ixpml]
            dEzx = dEzx / K[ixpml]
        end
        if iy <= ylayer1.ind   # y left layer [1:iy1]
            (; K, A, B) = ylayer1
            iypml = iy
            psiExy1[ix,iypml,iz] = B[iypml] * psiExy1[ix,iypml,iz] + A[iypml] * dExy
            psiEzy1[ix,iypml,iz] = B[iypml] * psiEzy1[ix,iypml,iz] + A[iypml] * dEzy
            Hx[ix,iy,iz] -= Mh * psiEzy1[ix,iypml,iz]
            Hz[ix,iy,iz] += Mh * psiExy1[ix,iypml,iz]
            dExy = dExy / K[iypml]
            dEzy = dEzy / K[iypml]
        end
        if iy >= ylayer2.ind   # y right layer [iy2:Ny]
            (; ind, K, A, B) = ylayer2
            iypml = iy - ind + 1
            psiExy2[ix,iypml,iz] = B[iypml] * psiExy2[ix,iypml,iz] + A[iypml] * dExy
            psiEzy2[ix,iypml,iz] = B[iypml] * psiEzy2[ix,iypml,iz] + A[iypml] * dEzy
            Hx[ix,iy,iz] -= Mh * psiEzy2[ix,iypml,iz]
            Hz[ix,iy,iz] += Mh * psiExy2[ix,iypml,iz]
            dExy = dExy / K[iypml]
            dEzy = dEzy / K[iypml]
        end
        if iz <= zlayer1.ind   # z left layer [1:iz1]
            (; K, A, B) = zlayer1
            izpml = iz
            psiExz1[ix,iy,izpml] = B[izpml] * psiExz1[ix,iy,izpml] + A[izpml] * dExz
            psiEyz1[ix,iy,izpml] = B[izpml] * psiEyz1[ix,iy,izpml] + A[izpml] * dEyz
            Hx[ix,iy,iz] += Mh * psiEyz1[ix,iy,izpml]
            Hy[ix,iy,iz] -= Mh * psiExz1[ix,iy,izpml]
            dExz = dExz / K[izpml]
            dEyz = dEyz / K[izpml]
        end
        if iz >= zlayer2.ind   # z right layer [iz2:Nz]
            (; ind, K, A, B) = zlayer2
            izpml = iz - ind + 1
            psiExz2[ix,iy,izpml] = B[izpml] * psiExz2[ix,iy,izpml] + A[izpml] * dExz
            psiEyz2[ix,iy,izpml] = B[izpml] * psiEyz2[ix,iy,izpml] + A[izpml] * dEyz
            Hx[ix,iy,iz] += Mh * psiEyz2[ix,iy,izpml]
            Hy[ix,iy,iz] -= Mh * psiExz2[ix,iy,izpml]
            dExz = dExz / K[izpml]
            dEyz = dEyz / K[izpml]
        end

        # update H .........................................................................
        Hx[ix,iy,iz] -= Mh * (dEzy - dEyz)
        Hy[ix,iy,iz] -= Mh * (dExz - dEzx)
        Hz[ix,iy,iz] -= Mh * (dEyx - dExy)
    end
end
function update_H!(model::Model{F}) where F <: Field3D
    (; Hx) = model.field
    backend = get_backend(Hx)
    ndrange = size(Hx)
    # In order to avoid the issue caused by the large size of the CUDA kernel parameters,
    # here we pass the parameters of the model explicitly:
    # https://discourse.julialang.org/t/passing-too-long-tuples-into-cuda-kernel-causes-an-error
    (; field, pml, material, Mh) = model
    update_H_kernel!(backend)(field, pml, material, Mh; ndrange)
    return nothing
end


# In order to avoid the issue caused by the large size of the CUDA kernel parameters,
# here we pass the parameters of the model explicitly:
# https://discourse.julialang.org/t/passing-too-long-tuples-into-cuda-kernel-causes-an-error
@kernel function update_E_kernel!(field::Field3D, pml, material, dt, Me0, Md10, Md20)
    (; grid, Hx, Hy, Hz, Dx, Dy, Dz, Ex, Ey, Ez) = field
    (; Nx, Ny, Nz, dx, dy, dz) = grid
    (; xlayer1, psiHyx1, psiHzx1, xlayer2, psiHyx2, psiHzx2,
       ylayer1, psiHxy1, psiHzy1, ylayer2, psiHxy2, psiHzy2,
       zlayer1, psiHxz1, psiHyz1, zlayer2, psiHxz2, psiHyz2) = pml
    (; geometry, isdispersion, iskerr, isplasma) = material

    ix, iy, iz = @index(Global, NTuple)

    @inbounds begin
        isgeometry = geometry[ix,iy,iz]

        if isgeometry
            # Me=1/(EPS0*eps), Md1=(1-sigma*dt/2)/(1+sigma*dt/2), Md2=dt/(1+sigma*dt/2)
            (; Me, Md1, Md2) = material
        else
            # Me=1/EPS0, Md1=1, Md2=dt
            Me, Md1, Md2 = Me0, Md10, Md20
        end

        # derivatives H ....................................................................
        ix == 1 ? ixm1 = Nx : ixm1 = ix - 1
        iy == 1 ? iym1 = Ny : iym1 = iy - 1
        iz == 1 ? izm1 = Nz : izm1 = iz - 1
        dHxy = (Hx[ix,iy,iz] - Hx[ix,iym1,iz]) / dy
        dHxz = (Hx[ix,iy,iz] - Hx[ix,iy,izm1]) / dz
        dHyx = (Hy[ix,iy,iz] - Hy[ixm1,iy,iz]) / dx
        dHyz = (Hy[ix,iy,iz] - Hy[ix,iy,izm1]) / dz
        dHzx = (Hz[ix,iy,iz] - Hz[ixm1,iy,iz]) / dx
        dHzy = (Hz[ix,iy,iz] - Hz[ix,iym1,iz]) / dy

        # apply CPML .......................................................................
        if ix <= xlayer1.ind   # x left layer [1:ix1]
            (; K, A, B) = xlayer1
            ixpml = ix
            psiHyx1[ixpml,iy,iz] = B[ixpml] * psiHyx1[ixpml,iy,iz] + A[ixpml] * dHyx
            psiHzx1[ixpml,iy,iz] = B[ixpml] * psiHzx1[ixpml,iy,iz] + A[ixpml] * dHzx
            Dy[ix,iy,iz] -= Md2 * psiHzx1[ixpml,iy,iz]
            Dz[ix,iy,iz] += Md2 * psiHyx1[ixpml,iy,iz]
            dHyx = dHyx / K[ixpml]
            dHzx = dHzx / K[ixpml]
        end
        if ix >= xlayer2.ind   # x right layer [ix2:Nx]
            (; ind, K, A, B) = xlayer2
            ixpml = ix - ind + 1
            psiHyx2[ixpml,iy,iz] = B[ixpml] * psiHyx2[ixpml,iy,iz] + A[ixpml] * dHyx
            psiHzx2[ixpml,iy,iz] = B[ixpml] * psiHzx2[ixpml,iy,iz] + A[ixpml] * dHzx
            Dy[ix,iy,iz] -= Md2 * psiHzx2[ixpml,iy,iz]
            Dz[ix,iy,iz] += Md2 * psiHyx2[ixpml,iy,iz]
            dHyx = dHyx / K[ixpml]
            dHzx = dHzx / K[ixpml]
        end
        if iy <= ylayer1.ind   # y left layer [1:iy1]
            (; K, A, B) = ylayer1
            iypml = iy
            psiHxy1[ix,iypml,iz] = B[iypml] * psiHxy1[ix,iypml,iz] + A[iypml] * dHxy
            psiHzy1[ix,iypml,iz] = B[iypml] * psiHzy1[ix,iypml,iz] + A[iypml] * dHzy
            Dx[ix,iy,iz] += Md2 * psiHzy1[ix,iypml,iz]
            Dz[ix,iy,iz] -= Md2 * psiHxy1[ix,iypml,iz]
            dHxy = dHxy / K[iypml]
            dHzy = dHzy / K[iypml]
        end
        if iy >= ylayer2.ind   # y right layer [iy2:Ny]
            (; ind, K, A, B) = ylayer2
            iypml = iy - ind + 1
            psiHxy2[ix,iypml,iz] = B[iypml] * psiHxy2[ix,iypml,iz] + A[iypml] * dHxy
            psiHzy2[ix,iypml,iz] = B[iypml] * psiHzy2[ix,iypml,iz] + A[iypml] * dHzy
            Dx[ix,iy,iz] += Md2 * psiHzy2[ix,iypml,iz]
            Dz[ix,iy,iz] -= Md2 * psiHxy2[ix,iypml,iz]
            dHxy = dHxy / K[iypml]
            dHzy = dHzy / K[iypml]
        end
        if iz <= zlayer1.ind   # z left layer [1:iz1]
            (; K, A, B) = zlayer1
            izpml = iz
            psiHxz1[ix,iy,izpml] = B[izpml] * psiHxz1[ix,iy,izpml] + A[izpml] * dHxz
            psiHyz1[ix,iy,izpml] = B[izpml] * psiHyz1[ix,iy,izpml] + A[izpml] * dHyz
            Dx[ix,iy,iz] -= Md2 * psiHyz1[ix,iy,izpml]
            Dy[ix,iy,iz] += Md2 * psiHxz1[ix,iy,izpml]
            dHxz = dHxz / K[izpml]
            dHyz = dHyz / K[izpml]
        end
        if iz >= zlayer2.ind   # z right layer [iz2:Nz]
            (; ind, K, A, B) = zlayer2
            izpml = iz - ind + 1
            psiHxz2[ix,iy,izpml] = B[izpml] * psiHxz2[ix,iy,izpml] + A[izpml] * dHxz
            psiHyz2[ix,iy,izpml] = B[izpml] * psiHyz2[ix,iy,izpml] + A[izpml] * dHyz
            Dx[ix,iy,iz] -= Md2 * psiHyz2[ix,iy,izpml]
            Dy[ix,iy,iz] += Md2 * psiHxz2[ix,iy,izpml]
            dHxz = dHxz / K[izpml]
            dHyz = dHyz / K[izpml]
        end

        # update D .........................................................................
        Dx[ix,iy,iz] = Md1 * Dx[ix,iy,iz] + Md2 * (dHzy - dHyz)
        Dy[ix,iy,iz] = Md1 * Dy[ix,iy,iz] + Md2 * (dHxz - dHzx)
        Dz[ix,iy,iz] = Md1 * Dz[ix,iy,iz] + Md2 * (dHyx - dHxy)

        # materials ........................................................................
        sumPx = zero(eltype(Ex))
        sumPy = zero(eltype(Ey))
        sumPz = zero(eltype(Ez))

        # linear polarization:
        if isdispersion && isgeometry
            (; Aq, Bq, Cq,
               Px, oldPx1, oldPx2, Py, oldPy1, oldPy2, Pz, oldPz1, oldPz2) = material
            Nq = size(Px, 1)
            for iq=1:Nq
                oldPx2[iq,ix,iy,iz] = oldPx1[iq,ix,iy,iz]
                oldPx1[iq,ix,iy,iz] = Px[iq,ix,iy,iz]
                Px[iq,ix,iy,iz] = Aq[iq] * Px[iq,ix,iy,iz] +
                                  Bq[iq] * oldPx2[iq,ix,iy,iz] +
                                  Cq[iq] * Ex[ix,iy,iz]
                oldPy2[iq,ix,iy,iz] = oldPy1[iq,ix,iy,iz]
                oldPy1[iq,ix,iy,iz] = Py[iq,ix,iy,iz]
                Py[iq,ix,iy,iz] = Aq[iq] * Py[iq,ix,iy,iz] +
                                  Bq[iq] * oldPy2[iq,ix,iy,iz] +
                                  Cq[iq] * Ey[ix,iy,iz]
                oldPz2[iq,ix,iy,iz] = oldPz1[iq,ix,iy,iz]
                oldPz1[iq,ix,iy,iz] = Pz[iq,ix,iy,iz]
                Pz[iq,ix,iy,iz] = Aq[iq] * Pz[iq,ix,iy,iz] +
                                  Bq[iq] * oldPz2[iq,ix,iy,iz] +
                                  Cq[iq] * Ez[ix,iy,iz]
                sumPx += Px[iq,ix,iy,iz]
                sumPy += Py[iq,ix,iy,iz]
                sumPz += Pz[iq,ix,iy,iz]
            end
        end

        # plasma:
        if isplasma && isgeometry
            (; ionrate, Rava, ksi, rho0, rho, drho, Ap, Bp, Cp,
               Ppx, oldPpx1, oldPpx2, Ppy, oldPpy1, oldPpy2, Ppz, oldPpz1, oldPpz2,
               Ma, Pax, Pay, Paz) = material

            E2 = abs2(Ex[ix,iy,iz]) + abs2(Ey[ix,iy,iz]) + abs2(Ez[ix,iy,iz])
            II = ksi * E2   # intensity

            # plasma current:
            oldPpx2[ix,iy,iz] = oldPpx1[ix,iy,iz]
            oldPpx1[ix,iy,iz] = Ppx[ix,iy,iz]
            Ppx[ix,iy,iz] = Ap * Ppx[ix,iy,iz] +
                            Bp * oldPpx2[ix,iy,iz] +
                            Cp * rho[ix,iy,iz]*rho0 * Ex[ix,iy,iz]
            oldPpy2[ix,iy,iz] = oldPpy1[ix,iy,iz]
            oldPpy1[ix,iy,iz] = Ppy[ix,iy,iz]
            Ppy[ix,iy,iz] = Ap * Ppy[ix,iy,iz] +
                            Bp * oldPpy2[ix,iy,iz] +
                            Cp * rho[ix,iy,iz]*rho0 * Ey[ix,iy,iz]
            oldPpz2[ix,iy,iz] = oldPpz1[ix,iy,iz]
            oldPpz1[ix,iy,iz] = Ppz[ix,iy,iz]
            Ppz[ix,iy,iz] = Ap * Ppz[ix,iy,iz] +
                            Bp * oldPpz2[ix,iy,iz] +
                            Cp * rho[ix,iy,iz]*rho0 * Ez[ix,iy,iz]
            sumPx += Ppx[ix,iy,iz]
            sumPy += Ppy[ix,iy,iz]
            sumPz += Ppz[ix,iy,iz]

            # multi-photon ionization losses:
            if E2 >= eps(one(E2))
                invE2 = 1 / E2
            else
                invE2 = zero(E2)
            end
            Pax[ix,iy,iz] += Ma * drho[ix,iy,iz]*rho0 * Ex[ix,iy,iz] * invE2
            Pay[ix,iy,iz] += Ma * drho[ix,iy,iz]*rho0 * Ey[ix,iy,iz] * invE2
            Paz[ix,iy,iz] += Ma * drho[ix,iy,iz]*rho0 * Ez[ix,iy,iz] * invE2
            sumPx += Pax[ix,iy,iz]
            sumPy += Pay[ix,iy,iz]
            sumPz += Paz[ix,iy,iz]

            # electron density:
            R1 = ionrate(II)
            R2 = Rava * E2
            if R2 == 0
                rho[ix,iy,iz] = 1 - (1 - rho[ix,iy,iz]) * exp(-R1 * dt)
            else
                R12 = R1 - R2
                rho[ix,iy,iz] = R1/R12*1 - (R1/R12*1 - rho[ix,iy,iz]) * exp(-R12 * dt)
            end
            drho[ix,iy,iz] = R1 * (1 - rho[ix,iy,iz])
        end

        # update E .........................................................................
        DmPx = Dx[ix,iy,iz] - sumPx
        DmPy = Dy[ix,iy,iz] - sumPy
        DmPz = Dz[ix,iy,iz] - sumPz

        if iskerr && isgeometry
            (; Mk2, Mk3) = material   # Mk2=EPS0*chi2, Mk3=EPS0*chi3

            # Kerr by Meep [A.F. Oskooi, Comput. Phys. Commun., 181, 687 (2010)]
            Ex[ix,iy,iz] = (1 + 1*Mk2 * Me^2 * DmPx + 2*Mk3 * Me^3 * DmPx^2) /
                           (1 + 2*Mk2 * Me^2 * DmPx + 3*Mk3 * Me^3 * DmPx^2) * DmPx * Me
            Ey[ix,iy,iz] = (1 + 1*Mk2 * Me^2 * DmPy + 2*Mk3 * Me^3 * DmPy^2) /
                           (1 + 2*Mk2 * Me^2 * DmPy + 3*Mk3 * Me^3 * DmPy^2) * DmPy * Me
            Ez[ix,iy,iz] = (1 + 1*Mk2 * Me^2 * DmPz + 2*Mk3 * Me^3 * DmPz^2) /
                           (1 + 2*Mk2 * Me^2 * DmPz + 3*Mk3 * Me^3 * DmPz^2) * DmPz * Me
        else
            Ex[ix,iy,iz] = DmPx * Me
            Ey[ix,iy,iz] = DmPy * Me
            Ez[ix,iy,iz] = DmPz * Me
        end

        # update E explicit:
        # (; dt) = model
        # Ex[ix,iy,iz] += dt * Me * ((dHzy - dHyz) + (psiHzy[ix,iy,iz] - psiHyz[ix,iy,iz]))
        # Ey[ix,iy,iz] += dt * Me * ((dHxz - dHzx) + (psiHxz[ix,iy,iz] - psiHzx[ix,iy,iz]))
        # Ez[ix,iy,iz] += dt * Me * ((dHyx - dHxy) + (psiHyx[ix,iy,iz] - psiHxy[ix,iy,iz]))
    end
end
function update_E!(model::Model{F}) where F <: Field3D
    (; Ex) = model.field
    backend = get_backend(Ex)
    ndrange = size(Ex)
    # In order to avoid the issue caused by the large size of the CUDA kernel parameters,
    # here we pass the parameters of the model explicitly:
    # https://discourse.julialang.org/t/passing-too-long-tuples-into-cuda-kernel-causes-an-error
    (; field, pml, material, dt, Me, Md1, Md2) = model
    update_E_kernel!(backend)(field, pml, material, dt, Me, Md1, Md2; ndrange)
    return nothing
end


# ******************************************************************************************
# Util
# ******************************************************************************************
function time_step(grid::Grid1D, CN)
    (; dz) = grid
    return CN / C0 / sqrt(1/dz^2)
end


function time_step(grid::Grid2D, CN)
    (; dx, dz) = grid
    return CN / C0 / sqrt(1/dx^2 + 1/dz^2)
end


function time_step(grid::Grid3D, CN)
    (; dx, dy, dz) = grid
    return CN / C0 / sqrt(1/dx^2 + 1/dy^2 + 1/dz^2)
end


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
