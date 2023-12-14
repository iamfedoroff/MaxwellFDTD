mutable struct Output{S, R, I, A1, A2}
    fname :: S
    # Fields data:
    isfields :: Bool
    itout :: Int
    Ntout :: Int
    tout :: R
    # View points data:
    isviewpoints :: Bool
    ipts :: I   # coordinates of the view points
    # Output variables data:
    Sa :: A1   # averaged poynting vector
    E2 :: A2   # averaged E^2
end


function write_output_variables(out, model)
    (; material) = model
    (; geometry, isplasma, rho, rho0) = material
    (; fname, Sa, E2) = out
    HDF5.h5open(fname, "r+") do fp
        fp["Sa"] = collect(Sa)   # averaged poynting vector
        if any(geometry)
            fp["E2"] = collect(E2)   # averaged E^2
        end
        if isplasma
            fp["rho_end"] = collect(rho) * rho0   # final plasma distribution
        end
    end
    return nothing
end


# ******************************************************************************************
# 1D: d/dx = d/dy = 0,   (Hy, Ex)
# ******************************************************************************************
function Output(
    model::Model{F}; fname="out.hdf", nstride=nothing, nframes=nothing, dtout=nothing,
    viewpoints=nothing,
) where F <: Field1D
    (; field, material, Nt, t) = model
    (; grid, Ex) = field
    (; geometry, isplasma) = material
    (; Nz, z) = grid

    if !isdir(dirname(fname))
        mkpath(dirname(fname))
    end

    if isnothing(nstride) && isnothing(nframes) && isnothing(dtout)
        isfields = false
        tout = nothing
        Ntout = 0
    else
        isfields = true
        tout = output_times(t, nstride, nframes, dtout)
        Ntout = length(tout)
    end
    itout = 1

    if isnothing(viewpoints)
        isviewpoints = false
        ipts = nothing
    else
        isviewpoints = true
        Np = length(viewpoints)
        ipts = Vector{CartesianIndices}(undef, Np)
        for (n, pt) in enumerate(viewpoints)
            zpt = pt
            izpt = argmin(abs.(z .- zpt))
            ipts[n] = CartesianIndices((izpt:izpt,))
        end
        ipts = (ipts...,)   # Vector -> Tuple
    end

    T = eltype(Ex)

    HDF5.h5open(fname, "w") do fp
        fp["z"] = collect(z)
        if any(geometry)
            fp["geometry"] = collect(geometry)
        end
        if isfields
            group = HDF5.create_group(fp, "fields")
            group["t"] = collect(tout)
            HDF5.create_dataset(group, "Hy", T, (Nz, Ntout))
            HDF5.create_dataset(group, "Ex", T, (Nz, Ntout))
            if isplasma
                HDF5.create_dataset(group, "rho", T, (Nz, Ntout))
            end
        end
        if isviewpoints
            fp["viewpoints/t"] = collect(t)
            for n=1:Np
                group = HDF5.create_group(fp, "viewpoints/$n")
                group["point"] = collect(promote(viewpoints[n]...))
                HDF5.create_dataset(group, "Hy", T, (Nt,))
                HDF5.create_dataset(group, "Ex", T, (Nt,))
                if isplasma
                    HDF5.create_dataset(group, "rho", T, (Nt,))
                end
            end
        end
    end

    Sa = zero(Ex)
    any(geometry) ? E2 = zero(Ex) : E2 = nothing

    return Output(fname, isfields, itout, Ntout, tout, isviewpoints, ipts, Sa, E2)
end


function write_fields(out, model::Model{F}) where F <: Field1D
    (; fname, isfields, itout) = out
    (; field, material) = model
    (; Hy, Ex) = field
    (; isplasma, rho, rho0) = material
    if isfields
        HDF5.h5open(fname, "r+") do fp
            group = fp["fields"]
            group["Hy"][:,itout] = collect(Hy)
            group["Ex"][:,itout] = collect(Ex)
            if isplasma
                group["rho"][:,itout] = collect(rho) * rho0
            end
        end
    end
    return nothing
end


function write_viewpoints(out, model::Model{F}, it) where F <: Field1D
    (; fname, isviewpoints, ipts) = out
    (; field, material) = model
    (; Hy, Ex) = field
    (; isplasma, rho, rho0) = material
    if isviewpoints
        HDF5.h5open(fname, "r+") do fp
            for (n, ipt) in enumerate(ipts)
                group = fp["viewpoints/$n"]
                group["Hy"][it] = collect(Hy[ipt])
                group["Ex"][it] = collect(Ex[ipt])
                if isplasma
                    group["rho"][it] = collect(rho[ipt]) * rho0
                end
            end
        end
    end
    return nothing
end


function calculate_output_variables!(out, model::Model{F}) where F <: Field1D
    (; Sa, E2) = out
    (; field, material, dt) = model
    (; Hy, Ex) = field
    (; geometry) = material
    @. Sa += sqrt((Ex*Hy)^2) * dt   # averaged poynting vector
    if any(geometry)
        @. E2 += Ex^2 * dt   # averaged E^2
    end
    return nothing
end


# ******************************************************************************************
# 2D: d/dy = 0,   (Hy, Ex, Ez)
# ******************************************************************************************
function Output(
    model::Model{F}; fname="out.hdf", nstride=nothing, nframes=nothing, dtout=nothing,
    viewpoints=nothing,
) where F <: Field2D
    (; field, material, Nt, t) = model
    (; grid, Ex) = field
    (; geometry, isplasma) = material
    (; Nx, Nz, x, z) = grid

    if !isdir(dirname(fname))
        mkpath(dirname(fname))
    end

    if isnothing(nstride) && isnothing(nframes) && isnothing(dtout)
        isfields = false
        tout = nothing
        Ntout = 0
    else
        isfields = true
        tout = output_times(t, nstride, nframes, dtout)
        Ntout = length(tout)
    end
    itout = 1

    if isnothing(viewpoints)
        isviewpoints = false
        ipts = nothing
    else
        isviewpoints = true
        Np = length(viewpoints)
        ipts = Vector{CartesianIndices}(undef, Np)
        for (n, pt) in enumerate(viewpoints)
            xpt, zpt = pt
            ixpt = argmin(abs.(x .- xpt))
            izpt = argmin(abs.(z .- zpt))
            ipts[n] = CartesianIndices((ixpt:ixpt, izpt:izpt))
        end
        ipts = (ipts...,)   # Vector -> Tuple
    end

    T = eltype(Ex)

    HDF5.h5open(fname, "w") do fp
        fp["x"] = collect(x)
        fp["z"] = collect(z)
        if any(geometry)
            fp["geometry"] = collect(geometry)
        end
        if isfields
            group = HDF5.create_group(fp, "fields")
            group["t"] = collect(tout)
            HDF5.create_dataset(group, "Hy", T, (Nx, Nz, Ntout))
            HDF5.create_dataset(group, "Ex", T, (Nx, Nz, Ntout))
            HDF5.create_dataset(group, "Ez", T, (Nx, Nz, Ntout))
            if isplasma
                HDF5.create_dataset(group, "rho", T, (Nx, Nz, Ntout))
            end
        end
        if isviewpoints
            fp["viewpoints/t"] = collect(t)
            for n=1:Np
                group = HDF5.create_group(fp, "viewpoints/$n")
                group["point"] = collect(promote(viewpoints[n]...))
                HDF5.create_dataset(group, "Hy", T, (Nt,))
                HDF5.create_dataset(group, "Ex", T, (Nt,))
                HDF5.create_dataset(group, "Ez", T, (Nt,))
                if isplasma
                    HDF5.create_dataset(group, "rho", T, (Nt,))
                end
            end
        end
    end

    Sa = zero(Ex)
    any(geometry) ? E2 = zero(Ex) : E2 = nothing

    return Output(fname, isfields, itout, Ntout, tout, isviewpoints, ipts, Sa, E2)
end


function write_fields(out, model::Model{F}) where F <: Field2D
    (; fname, isfields, itout) = out
    (; field, material) = model
    (; Hy, Ex, Ez) = field
    (; isplasma, rho, rho0) = material
    if isfields
        HDF5.h5open(fname, "r+") do fp
            group = fp["fields"]
            group["Hy"][:,:,itout] = collect(Hy)
            group["Ex"][:,:,itout] = collect(Ex)
            group["Ez"][:,:,itout] = collect(Ez)
            if isplasma
                group["rho"][:,:,itout] = collect(rho) * rho0
            end
        end
    end
    return nothing
end


function write_viewpoints(out, model::Model{F}, it) where F <: Field2D
    (; fname, isviewpoints, ipts) = out
    (; field, material) = model
    (; Hy, Ex, Ez) = field
    (; isplasma, rho, rho0) = material
    if isviewpoints
        HDF5.h5open(fname, "r+") do fp
            for (n, ipt) in enumerate(ipts)
                group = fp["viewpoints/$n"]
                group["Hy"][it] = collect(Hy[ipt])
                group["Ex"][it] = collect(Ex[ipt])
                group["Ez"][it] = collect(Ez[ipt])
                if isplasma
                    group["rho"][it] = collect(rho[ipt]) * rho0
                end
            end
        end
    end
    return nothing
end


function calculate_output_variables!(out, model::Model{F}) where F <: Field2D
    (; Sa, E2) = out
    (; field, material, dt) = model
    (; Hy, Ex, Ez) = field
    (; geometry) = material
    @. Sa += sqrt((-Ez*Hy)^2 + (Ex*Hy)^2) * dt   # averaged poynting vector
    if any(geometry)
        @. E2 += (Ex^2 + Ez^2) * dt   # average E^2
    end
    return nothing
end


# ******************************************************************************************
# 3D
# ******************************************************************************************
function Output(
    model::Model{F}; fname="out.hdf", nstride=nothing, nframes=nothing, dtout=nothing,
    viewpoints=nothing,
) where F <: Field3D
    (; field, material, Nt, t) = model
    (; grid, Ex) = field
    (; geometry, isplasma) = material
    (; Nx, Ny, Nz, x, y, z) = grid

    if !isdir(dirname(fname))
        mkpath(dirname(fname))
    end

    if isnothing(nstride) && isnothing(nframes) && isnothing(dtout)
        isfields = false
        tout = nothing
        Ntout = 0
    else
        isfields = true
        tout = output_times(t, nstride, nframes, dtout)
        Ntout = length(tout)
    end
    itout = 1

    if isnothing(viewpoints)
        isviewpoints = false
        ipts = nothing
    else
        isviewpoints = true
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
    end

    T = eltype(Ex)

    HDF5.h5open(fname, "w") do fp
        fp["x"] = collect(x)
        fp["y"] = collect(y)
        fp["z"] = collect(z)
        if any(geometry)
            fp["geometry"] = collect(geometry)
        end
        if isfields
            group = HDF5.create_group(fp, "fields")
            group["t"] = collect(tout)
            HDF5.create_dataset(group, "Hx", T, (Nx, Ny, Nz, Ntout))
            HDF5.create_dataset(group, "Hy", T, (Nx, Ny, Nz, Ntout))
            HDF5.create_dataset(group, "Hz", T, (Nx, Ny, Nz, Ntout))
            HDF5.create_dataset(group, "Ex", T, (Nx, Ny, Nz, Ntout))
            HDF5.create_dataset(group, "Ey", T, (Nx, Ny, Nz, Ntout))
            HDF5.create_dataset(group, "Ez", T, (Nx, Ny, Nz, Ntout))
            if isplasma
                HDF5.create_dataset(group, "rho", T, (Nx, Ny, Nz, Ntout))
            end
        end
        if isviewpoints
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
                if isplasma
                    HDF5.create_dataset(group, "rho", T, (Nt,))
                end
            end
        end
    end

    Sa = zero(Ex)
    any(geometry) ? E2 = zero(Ex) : E2 = nothing

    return Output(fname, isfields, itout, Ntout, tout, isviewpoints, ipts, Sa, E2)
end


function write_fields(out, model::Model{F}) where F <: Field3D
    (; fname, isfields, itout) = out
    (; field, material) = model
    (; Hx, Hy, Hz, Ex, Ey, Ez) = field
    (; isplasma, rho, rho0) = material
    if isfields
        HDF5.h5open(fname, "r+") do fp
            group = fp["fields"]
            group["Hx"][:,:,:,itout] = collect(Hx)
            group["Hy"][:,:,:,itout] = collect(Hy)
            group["Hz"][:,:,:,itout] = collect(Hz)
            group["Ex"][:,:,:,itout] = collect(Ex)
            group["Ey"][:,:,:,itout] = collect(Ey)
            group["Ez"][:,:,:,itout] = collect(Ez)
            if isplasma
                group["rho"][:,:,:,itout] = collect(rho) * rho0
            end
        end
    end
    return nothing
end


function write_viewpoints(out, model::Model{F}, it) where F <: Field3D
    (; fname, isviewpoints, ipts) = out
    (; field, material) = model
    (; Hx, Hy, Hz, Ex, Ey, Ez) = field
    (; isplasma, rho, rho0) = material
    if isviewpoints
        HDF5.h5open(fname, "r+") do fp
            for (n, ipt) in enumerate(ipts)
                group = fp["viewpoints/$n"]
                group["Hx"][it] = collect(Hx[ipt])
                group["Hy"][it] = collect(Hy[ipt])
                group["Hz"][it] = collect(Hz[ipt])
                group["Ex"][it] = collect(Ex[ipt])
                group["Ey"][it] = collect(Ey[ipt])
                group["Ez"][it] = collect(Ez[ipt])
                if isplasma
                    group["rho"][it] = collect(rho[ipt]) * rho0
                end
            end
        end
    end
    return nothing
end


function calculate_output_variables!(out, model::Model{F}) where F <: Field3D
    (; Sa, E2) = out
    (; field, material, dt) = model
    (; Hx, Hy, Hz, Ex, Ey, Ez) = field
    (; geometry) = material
    # averaged poynting vector:
    @. Sa += sqrt((Ey*Hz - Ez*Hy)^2 + (Ez*Hx - Ex*Hz)^2 + (Ex*Hy - Ey*Hx)^2) * dt
    if any(geometry)
        @. E2 += (Ex^2 + Ey^2 + Ez^2) * dt   # averaged E^2
    end
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
