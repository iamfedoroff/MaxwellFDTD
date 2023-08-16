mutable struct Output{S, R, I, A}
    fname :: S
    itout :: Int
    Ntout :: Int
    tout :: R
    ipts :: I   # coordinates of the observation points
    Sa :: A   # averaged poynting vector
end


function write_output_variables(out, model)
    (; material) = model
    (; plasma, rho, rho0) = material
    (; fname, Sa) = out
    HDF5.h5open(fname, "r+") do fp
        fp["Sa"] = collect(Sa)   # averaged poynting vector
        if plasma
            fp["rho_end"] = collect(rho) * rho0   # final plasma distribution
        end
    end
    return nothing
end


# ******************************************************************************************
# 1D: d/dx = d/dy = 0,   (Hy, Ex)
# ******************************************************************************************
function Output(
    model::Model{F}; fname=nothing, nstride=nothing, nframes=nothing, dtout=nothing,
    viewpoints=nothing,
) where F <: Field1D
    (; field, material, Nt, t) = model
    (; grid, Ex) = field
    (; plasma) = material
    (; Nz, z) = grid

    tout = output_times(t, nstride, nframes, dtout)
    Ntout = length(tout)
    itout = 1

    if isnothing(fname)
        fname = "results/1d_out.hdf"
    end

    if !isdir(dirname(fname))
        mkpath(dirname(fname))
    end

    if !isnothing(viewpoints)
        Np = length(viewpoints)
        ipts = Vector{CartesianIndices}(undef, Np)
        for (n, pt) in enumerate(viewpoints)
            zpt = pt
            izpt = argmin(abs.(z .- zpt))
            ipts[n] = CartesianIndices((izpt:izpt,))
        end
        ipts = (ipts...,)   # Vector -> Tuple
    else
        ipts = nothing
    end

    T = eltype(Ex)

    HDF5.h5open(fname, "w") do fp
        fp["z"] = collect(z)
        fp["t"] = collect(tout)
        HDF5.create_dataset(fp, "Hy", T, (Nz, Ntout))
        HDF5.create_dataset(fp, "Ex", T, (Nz, Ntout))
        if plasma
            HDF5.create_dataset(fp, "rho", T, (Nz, Ntout))
        end
        if !isnothing(viewpoints)
            fp["viewpoints/t"] = collect(t)
            for n=1:Np
                group = HDF5.create_group(fp, "viewpoints/$n")
                group["point"] = collect(promote(viewpoints[n]...))
                HDF5.create_dataset(group, "Hy", T, (Nt,))
                HDF5.create_dataset(group, "Ex", T, (Nt,))
                if plasma
                    HDF5.create_dataset(group, "rho", T, (Nt,))
                end
            end
        end
    end

    Sa = zero(Ex)

    return Output(fname, itout, Ntout, tout, ipts, Sa)
end


function write_fields(out, model::Model{F}) where F <: Field1D
    (; field, material) = model
    (; Hy, Ex) = field
    (; plasma, rho, rho0) = material
    (; fname, itout) = out
    HDF5.h5open(fname, "r+") do fp
        fp["Hy"][:,itout] = collect(Hy)
        fp["Ex"][:,itout] = collect(Ex)
        if plasma
            fp["rho"][:,itout] = collect(rho) * rho0
        end
    end
    return nothing
end


function write_viewpoints(out, model::Model{F}, it) where F <: Field1D
    (; fname, ipts) = out
    (; field, material) = model
    (; Hy, Ex) = field
    (; plasma, rho, rho0) = material
    if !isnothing(ipts)
        HDF5.h5open(fname, "r+") do fp
            for (n, ipt) in enumerate(ipts)
                group = fp["viewpoints/$n"]
                group["Hy"][it] = collect(Hy[ipt])
                group["Ex"][it] = collect(Ex[ipt])
                if plasma
                    group["rho"][it] = collect(rho[ipt]) * rho0
                end
            end
        end
    end
    return nothing
end


function calculate_output_variables!(out, model::Model{F}) where F <: Field1D
    (; Sa) = out
    (; field, dt) = model
    (; Hy, Ex) = field
    @. Sa += sqrt((Ex*Hy)^2) * dt   # averaged poynting vector
    return nothing
end


# ******************************************************************************************
# 2D: d/dy = 0,   (Hy, Ex, Ez)
# ******************************************************************************************
function Output(
    model::Model{F}; fname=nothing, nstride=nothing, nframes=nothing, dtout=nothing,
    viewpoints=nothing,
) where F <: Field2D
    (; field, material, Nt, t) = model
    (; grid, Ex) = field
    (; plasma) = material
    (; Nx, Nz, x, z) = grid

    tout = output_times(t, nstride, nframes, dtout)
    Ntout = length(tout)
    itout = 1

    if isnothing(fname)
        fname = "results/2d_out.hdf"
    end

    if !isdir(dirname(fname))
        mkpath(dirname(fname))
    end

    if !isnothing(viewpoints)
        Np = length(viewpoints)
        ipts = Vector{CartesianIndices}(undef, Np)
        for (n, pt) in enumerate(viewpoints)
            xpt, zpt = pt
            ixpt = argmin(abs.(x .- xpt))
            izpt = argmin(abs.(z .- zpt))
            ipts[n] = CartesianIndices((ixpt:ixpt, izpt:izpt))
        end
        ipts = (ipts...,)   # Vector -> Tuple
    else
        ipts = nothing
    end

    T = eltype(Ex)

    HDF5.h5open(fname, "w") do fp
        fp["x"] = collect(x)
        fp["z"] = collect(z)
        fp["t"] = collect(tout)
        HDF5.create_dataset(fp, "Hy", T, (Nx, Nz, Ntout))
        HDF5.create_dataset(fp, "Ex", T, (Nx, Nz, Ntout))
        HDF5.create_dataset(fp, "Ez", T, (Nx, Nz, Ntout))
        if plasma
            HDF5.create_dataset(fp, "rho", T, (Nx, Nz, Ntout))
        end
        if !isnothing(viewpoints)
            fp["viewpoints/t"] = collect(t)
            for n=1:Np
                group = HDF5.create_group(fp, "viewpoints/$n")
                group["point"] = collect(promote(viewpoints[n]...))
                HDF5.create_dataset(group, "Hy", T, (Nt,))
                HDF5.create_dataset(group, "Ex", T, (Nt,))
                HDF5.create_dataset(group, "Ez", T, (Nt,))
                if plasma
                    HDF5.create_dataset(group, "rho", T, (Nt,))
                end
            end
        end
    end

    Sa = zero(Ex)

    return Output(fname, itout, Ntout, tout, ipts, Sa)
end


function write_fields(out, model::Model{F}) where F <: Field2D
    (; field, material) = model
    (; Hy, Ex, Ez) = field
    (; plasma, rho, rho0) = material
    (; fname, itout) = out
    HDF5.h5open(fname, "r+") do fp
        fp["Hy"][:,:,itout] = collect(Hy)
        fp["Ex"][:,:,itout] = collect(Ex)
        fp["Ez"][:,:,itout] = collect(Ez)
        if plasma
            fp["rho"][:,:,itout] = collect(rho) * rho0
        end
    end
    return nothing
end


function write_viewpoints(out, model::Model{F}, it) where F <: Field2D
    (; fname, ipts) = out
    (; field, material) = model
    (; Hy, Ex, Ez) = field
    (; plasma, rho, rho0) = material
    if !isnothing(ipts)
        HDF5.h5open(fname, "r+") do fp
            for (n, ipt) in enumerate(ipts)
                group = fp["viewpoints/$n"]
                group["Hy"][it] = collect(Hy[ipt])
                group["Ex"][it] = collect(Ex[ipt])
                group["Ez"][it] = collect(Ez[ipt])
                if plasma
                    group["rho"][it] = collect(rho[ipt]) * rho0
                end
            end
        end
    end
    return nothing
end


function calculate_output_variables!(out, model::Model{F}) where F <: Field2D
    (; Sa) = out
    (; field, dt) = model
    (; Hy, Ex, Ez) = field
    @. Sa += sqrt((-Ez*Hy)^2 + (Ex*Hy)^2) * dt   # averaged poynting vector
    return nothing
end


# ******************************************************************************************
# 3D
# ******************************************************************************************
function Output(
    model::Model{F}; fname=nothing, nstride=nothing, nframes=nothing, dtout=nothing,
    viewpoints=nothing,
) where F <: Field3D
    (; field, material, Nt, t) = model
    (; grid, Ex) = field
    (; plasma) = material
    (; Nx, Ny, Nz, x, y, z) = grid

    tout = output_times(t, nstride, nframes, dtout)
    Ntout = length(tout)
    itout = 1

    if isnothing(fname)
        fname = "results/3d_out.hdf"
    end

    if !isdir(dirname(fname))
        mkpath(dirname(fname))
    end

    if !isnothing(viewpoints)
        Np = length(viewpoints)
        ipts = Vector{CartesianIndices}(undef, Np)
        for (n, pt) in enumerate(viewpoints)
            xpt, ypt, zpt = pt
            ixpt = argmin(abs.(x .- xpt))
            iypt = argmin(abs.(y .- ypt))
            izpt = argmin(abs.(z .- zpt))
            ipts[n] = CartesianIndices((ixpt:ixpt, iypt:iypt, izpt:izpt))
        end
        ipts = (ipts...,)   # Vector -> Tuple
    else
        ipts = nothing
    end

    T = eltype(Ex)

    HDF5.h5open(fname, "w") do fp
        fp["x"] = collect(x)
        fp["y"] = collect(y)
        fp["z"] = collect(z)
        fp["t"] = collect(tout)
        HDF5.create_dataset(fp, "Hx", T, (Nx, Ny, Nz, Ntout))
        HDF5.create_dataset(fp, "Hy", T, (Nx, Ny, Nz, Ntout))
        HDF5.create_dataset(fp, "Hz", T, (Nx, Ny, Nz, Ntout))
        HDF5.create_dataset(fp, "Ex", T, (Nx, Ny, Nz, Ntout))
        HDF5.create_dataset(fp, "Ey", T, (Nx, Ny, Nz, Ntout))
        HDF5.create_dataset(fp, "Ez", T, (Nx, Ny, Nz, Ntout))
        if plasma
            HDF5.create_dataset(fp, "rho", T, (Nx, Ny, Nz, Ntout))
        end
        if !isnothing(viewpoints)
            fp["viewpoints/t"] = collect(t)
            for n=1:Np
                group = HDF5.create_group(fp, "viewpoints/$n")
                group["point"] = collect(promote(viewpoints[n]...))
                HDF5.create_dataset(group, "Hx", T, (Nt,))
                HDF5.create_dataset(group, "Hy", T, (Nt,))
                HDF5.create_dataset(group, "Hz", T, (Nt,))
                HDF5.create_dataset(group, "Ex", T, (Nt,))
                HDF5.create_dataset(group, "Ey", T, (Nt,))
                HDF5.create_dataset(group, "Ez", T, (Nt,))
                if plasma
                    HDF5.create_dataset(group, "rho", T, (Nt,))
                end
            end
        end
    end

    Sa = zero(Ex)

    return Output(fname, itout, Ntout, tout, ipts, Sa)
end


function write_fields(out, model::Model{F}) where F <: Field3D
    (; field, material) = model
    (; Hx, Hy, Hz, Ex, Ey, Ez) = field
    (; plasma, rho, rho0) = material
    (; fname, itout) = out
    HDF5.h5open(fname, "r+") do fp
        fp["Hx"][:,:,:,itout] = collect(Hx)
        fp["Hy"][:,:,:,itout] = collect(Hy)
        fp["Hz"][:,:,:,itout] = collect(Hz)
        fp["Ex"][:,:,:,itout] = collect(Ex)
        fp["Ey"][:,:,:,itout] = collect(Ey)
        fp["Ez"][:,:,:,itout] = collect(Ez)
        if plasma
            fp["rho"][:,:,:,itout] = collect(rho) * rho0
        end
    end
    return nothing
end


function write_viewpoints(out, model::Model{F}, it) where F <: Field3D
    (; fname, ipts) = out
    (; field, material) = model
    (; Hx, Hy, Hz, Ex, Ey, Ez) = field
    (; plasma, rho, rho0) = material
    if !isnothing(ipts)
        HDF5.h5open(fname, "r+") do fp
            for (n, ipt) in enumerate(ipts)
                group = fp["viewpoints/$n"]
                group["Hx"][it] = collect(Hx[ipt])
                group["Hy"][it] = collect(Hy[ipt])
                group["Hz"][it] = collect(Hz[ipt])
                group["Ex"][it] = collect(Ex[ipt])
                group["Ey"][it] = collect(Ey[ipt])
                group["Ez"][it] = collect(Ez[ipt])
                if plasma
                    group["rho"][it] = collect(rho[ipt]) * rho0
                end
            end
        end
    end
    return nothing
end


function calculate_output_variables!(out, model::Model{F}) where F <: Field3D
    (; Sa) = out
    (; field, dt) = model
    (; Hx, Hy, Hz, Ex, Ey, Ez) = field
    # averaged poynting vector:
    @. Sa += sqrt((Ey*Hz - Ez*Hy)^2 + (Ez*Hx - Ex*Hz)^2 + (Ex*Hy - Ey*Hx)^2) * dt
    return nothing
end


# ******************************************************************************************
# Util
# ******************************************************************************************
function output_times(t, nstride, nframes, dtout)
    Nt = length(t)

    if isnothing(nstride) && isnothing(nframes) && isnothing(dtout)
        error("One of 'nstride', 'nframes', or 'dtout' should be specified.")
    elseif !isnothing(nstride) && isnothing(nframes) && isnothing(dtout)
        if nstride >= Nt
            nstride = Nt-1
        end
    elseif isnothing(nstride) && !isnothing(nframes) && isnothing(dtout)
        if nframes >= Nt
            nstride = 1
        else
            nstride = round(Int, Nt / nframes)
        end
    elseif isnothing(nstride) && isnothing(nframes) && !isnothing(dtout)
        tmax = t[end]
        if dtout >= tmax
            nstride = 1
        else
            nstride = round(Int, Nt / (t[end] / dtout))
        end
    else
        error("Only one of 'nstride', 'nframes', or 'dtout' can be specified.")
    end

    return t[1:nstride:end]
end
