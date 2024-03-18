mutable struct Output{S, R, C, M, A1, A2}
    fname :: S
    isgeometry :: Bool
    # Fields data:
    isfields :: Bool
    itout :: Int
    Ntout :: Int
    tout :: R
    components :: C
    # Monitors:
    ismonitors :: Bool
    monitors :: M
    # Integral variables:
    Sa :: A1   # averaged poynting vector
    E2 :: A2   # averaged E^2
end


function update_monitors!(out, model, it)
    (; ismonitors, monitors) = out
    (; field) = model
    if ismonitors
        for monitor in monitors
            update_monitor!(monitor, field, it)
        end
    end
    return nothing
end


function write_monitors(out)
    (; fname, ismonitors, monitors) = out
    if ismonitors
        HDF5.h5open(fname, "r+") do fp
            for (n, monitor) in enumerate(monitors)
                write_monitor(fp, n, monitor)
            end
        end
    end
    return nothing
end


function write_integral_variables(out, model)
    (; materials) = model
    (; isgeometry, fname, Sa, E2) = out
    HDF5.h5open(fname, "r+") do fp
        fp["Sa"] = collect(Sa)   # averaged poynting vector
        if isgeometry
            fp["E2"] = collect(E2)   # averaged E^2

            (; isplasma, rho, rho0) = materials[1]
            if isplasma
                fp["rho_end"] = collect(rho) * rho0   # final plasma distribution
            end
        end
    end
    return nothing
end


# ******************************************************************************************
# 1D: d/dx = d/dy = 0,   (Hy, Ex)
# ******************************************************************************************
function Output(
    model::Model{F}; fname="out.hdf", nstride=nothing, nframes=nothing, dtout=nothing,
    components=nothing, monitors=nothing,
) where F <: Field1D
    (; field, pml, geometry, materials, t) = model
    (; grid, Ex) = field
    (; Nz, z) = grid

    isgeometry = any(x -> x > 0, geometry)

    isplasma = any([material.isplasma for material in materials])
    if isplasma && length(materials) > 1
        @warn "The electron density is written only for the first material!"
    end

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

    if isnothing(components)
        components = (:Hy, :Ex)
    else
        components = Tuple(components)
        for comp in components
            if comp != :Hy && comp != :Ex
                error(
                    "You asked to output $(comp) field component, but 1D field has only " *
                    ":Hy and :Ex components."
                )
            end
        end
    end

    if isnothing(monitors)
        ismonitors = false
    else
        ismonitors = true
        monitors = monitors isa Monitor ? (monitors,) : monitors   # convert to array-like
        monitors = [Monitor(monitor, grid, t) for monitor in monitors]
    end

    T = eltype(Ex)

    ipml = [pml.zlayer1.ib, pml.zlayer2.ib]

    HDF5.h5open(fname, "w") do fp
        fp["z"] = collect(z)
        fp["t"] = collect(t)
        fp["pml"] = ipml
        if isgeometry
            fp["geometry"] = collect(geometry)
        end
        if isfields
            group = HDF5.create_group(fp, "fields")
            group["t"] = collect(tout)
            if :Hy in components
                HDF5.create_dataset(group, "Hy", T, (Nz, Ntout))
            end
            if :Ex in components
                HDF5.create_dataset(group, "Ex", T, (Nz, Ntout))
            end
            if isplasma
                HDF5.create_dataset(group, "rho", T, (Nz, Ntout))
            end
        end
    end

    Sa = zero(Ex)
    E2 = isgeometry ? zero(Ex) : nothing

    return Output(
        fname, isgeometry, isfields, itout, Ntout, tout, components, ismonitors, monitors,
        Sa, E2,
    )
end


function write_fields(out, model::Model{F}) where F <: Field1D
    (; fname, isgeometry, isfields, itout, components) = out
    (; field, materials) = model
    (; Hy, Ex) = field
    if isfields
        HDF5.h5open(fname, "r+") do fp
            group = fp["fields"]
            if :Hy in components
                group["Hy"][:,itout] = collect(Hy)
            end
            if :Ex in components
                group["Ex"][:,itout] = collect(Ex)
            end
            if isgeometry
                (; isplasma, rho, rho0) = materials[1]
                if isplasma
                    group["rho"][:,itout] = collect(rho) * rho0
                end
            end
        end
    end
    return nothing
end


function update_integral_variables!(out, model::Model{F}) where F <: Field1D
    (; isgeometry, Sa, E2) = out
    (; field, dt) = model
    (; Hy, Ex) = field
    @. Sa += sqrt((Ex*Hy)^2) * dt   # averaged poynting vector
    if isgeometry
        @. E2 += Ex^2 * dt   # averaged E^2
    end
    return nothing
end


# ******************************************************************************************
# 2D: d/dy = 0,   (Hy, Ex, Ez)
# ******************************************************************************************
function Output(
    model::Model{F}; fname="out.hdf", nstride=nothing, nframes=nothing, dtout=nothing,
    components=nothing, monitors=nothing,
) where F <: Field2D
    (; field, pml, geometry, materials, t) = model
    (; grid, Ex) = field
    (; Nx, Nz, x, z) = grid

    isgeometry = any(x -> x > 0, geometry)

    isplasma = any([material.isplasma for material in materials])
    if isplasma && length(materials) > 1
        @warn "The electron density is written only for the first material!"
    end

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

    if isnothing(components)
        components = (:Hy, :Ex, :Ez)
    else
        components = Tuple(components)
        for comp in components
            if comp != :Hy && comp != :Ex && comp != :Ez
                error(
                    "You asked to output $(comp) field component, but 2D field has only " *
                    ":Hy, :Ex, and :Ez components."
                )
            end
        end
    end

    if isnothing(monitors)
        ismonitors = false
    else
        ismonitors = true
        monitors = monitors isa Monitor ? (monitors,) : monitors   # convert to array-like
        monitors = [Monitor(monitor, grid, t) for monitor in monitors]
    end

    T = eltype(Ex)

    ipml = [
        pml.xlayer1.ib, pml.xlayer2.ib,
        pml.zlayer1.ib, pml.zlayer2.ib,
    ]

    HDF5.h5open(fname, "w") do fp
        fp["x"] = collect(x)
        fp["z"] = collect(z)
        fp["t"] = collect(t)
        fp["pml"] = ipml
        if isgeometry
            fp["geometry"] = collect(geometry)
        end
        if isfields
            group = HDF5.create_group(fp, "fields")
            group["t"] = collect(tout)
            if :Hy in components
                HDF5.create_dataset(group, "Hy", T, (Nx, Nz, Ntout))
            end
            if :Ex in components
                HDF5.create_dataset(group, "Ex", T, (Nx, Nz, Ntout))
            end
            if :Ez in components
                HDF5.create_dataset(group, "Ez", T, (Nx, Nz, Ntout))
            end
            if isplasma
                HDF5.create_dataset(group, "rho", T, (Nx, Nz, Ntout))
            end
        end
    end

    Sa = zero(Ex)
    E2 = isgeometry ? zero(Ex) : nothing

    return Output(
        fname, isgeometry, isfields, itout, Ntout, tout, components, ismonitors, monitors,
        Sa, E2,
    )
end


function write_fields(out, model::Model{F}) where F <: Field2D
    (; fname, isgeometry, isfields, itout, components) = out
    (; field, materials) = model
    (; Hy, Ex, Ez) = field
    if isfields
        HDF5.h5open(fname, "r+") do fp
            group = fp["fields"]
            if :Hy in components
                group["Hy"][:,:,itout] = collect(Hy)
            end
            if :Ex in components
                group["Ex"][:,:,itout] = collect(Ex)
            end
            if :Ez in components
                group["Ez"][:,:,itout] = collect(Ez)
            end
            if isgeometry
                (; isplasma, rho, rho0) = materials[1]
                if isplasma
                    group["rho"][:,:,itout] = collect(rho) * rho0
                end
            end
        end
    end
    return nothing
end


function update_integral_variables!(out, model::Model{F}) where F <: Field2D
    (; isgeometry, Sa, E2) = out
    (; field, dt) = model
    (; Hy, Ex, Ez) = field
    @. Sa += sqrt((-Ez*Hy)^2 + (Ex*Hy)^2) * dt   # averaged poynting vector
    if isgeometry
        @. E2 += (Ex^2 + Ez^2) * dt   # average E^2
    end
    return nothing
end


# ******************************************************************************************
# 3D
# ******************************************************************************************
function Output(
    model::Model{F}; fname="out.hdf", nstride=nothing, nframes=nothing, dtout=nothing,
    components=nothing, monitors=nothing,
) where F <: Field3D
    (; field, pml, geometry, materials, t) = model
    (; grid, Ex) = field
    (; Nx, Ny, Nz, x, y, z) = grid

    isgeometry = any(x -> x > 0, geometry)

    isplasma = any([material.isplasma for material in materials])
    if isplasma && length(materials) > 1
        @warn "The electron density is written only for the first material!"
    end

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

    if isnothing(components)
        components = (:Hx, :Hy, :Hz, :Ex, :Ey, :Ez)
    else
        components = Tuple(components)
        for comp in components
            if comp != :Hx && comp != :Hy && comp != :Hz &&
               comp != :Ex && comp != :Ey && comp != :Ez
                error(
                    "You asked to output $(comp) field component, but 3D field has only " *
                    ":Hx, :Hy, :Hz, :Ex, :Ey, and :Ez components."
                )
            end
        end
    end

    if isnothing(monitors)
        ismonitors = false
    else
        ismonitors = true
        monitors = monitors isa Monitor ? (monitors,) : monitors   # convert to array-like
        monitors = [Monitor(monitor, grid, t) for monitor in monitors]
    end

    T = eltype(Ex)

    ipml = [
        pml.xlayer1.ib, pml.xlayer2.ib,
        pml.ylayer1.ib, pml.ylayer2.ib,
        pml.zlayer1.ib, pml.zlayer2.ib,
    ]

    HDF5.h5open(fname, "w") do fp
        fp["x"] = collect(x)
        fp["y"] = collect(y)
        fp["z"] = collect(z)
        fp["t"] = collect(t)
        fp["pml"] = ipml
        if isgeometry
            fp["geometry"] = collect(geometry)
        end
        if isfields
            group = HDF5.create_group(fp, "fields")
            group["t"] = collect(tout)
            if :Hx in components
                HDF5.create_dataset(group, "Hx", T, (Nx, Ny, Nz, Ntout))
            end
            if :Hy in components
                HDF5.create_dataset(group, "Hy", T, (Nx, Ny, Nz, Ntout))
            end
            if :Hz in components
                HDF5.create_dataset(group, "Hz", T, (Nx, Ny, Nz, Ntout))
            end
            if :Ex in components
                HDF5.create_dataset(group, "Ex", T, (Nx, Ny, Nz, Ntout))
            end
            if :Ey in components
                HDF5.create_dataset(group, "Ey", T, (Nx, Ny, Nz, Ntout))
            end
            if :Ez in components
                HDF5.create_dataset(group, "Ez", T, (Nx, Ny, Nz, Ntout))
            end
            if isplasma
                HDF5.create_dataset(group, "rho", T, (Nx, Ny, Nz, Ntout))
            end
        end
    end

    Sa = zero(Ex)
    E2 = isgeometry ? zero(Ex) : nothing

    return Output(
        fname, isgeometry, isfields, itout, Ntout, tout, components, ismonitors, monitors,
        Sa, E2,
    )
end


function write_fields(out, model::Model{F}) where F <: Field3D
    (; fname, isgeometry, isfields, itout, components) = out
    (; field, materials) = model
    (; Hx, Hy, Hz, Ex, Ey, Ez) = field
    if isfields
        HDF5.h5open(fname, "r+") do fp
            group = fp["fields"]
            if :Hx in components
                group["Hx"][:,:,:,itout] = collect(Hx)
            end
            if :Hy in components
                group["Hy"][:,:,:,itout] = collect(Hy)
            end
            if :Hz in components
                group["Hz"][:,:,:,itout] = collect(Hz)
            end
            if :Ex in components
                group["Ex"][:,:,:,itout] = collect(Ex)
            end
            if :Ey in components
                group["Ey"][:,:,:,itout] = collect(Ey)
            end
            if :Ez in components
                group["Ez"][:,:,:,itout] = collect(Ez)
            end
            if isgeometry
                (; isplasma, rho, rho0) = materials[1]
                if isplasma
                    group["rho"][:,:,:,itout] = collect(rho) * rho0
                end
            end
        end
    end
    return nothing
end


function update_integral_variables!(out, model::Model{F}) where F <: Field3D
    (; isgeometry, Sa, E2) = out
    (; field, dt) = model
    (; Hx, Hy, Hz, Ex, Ey, Ez) = field
    # averaged poynting vector:
    @. Sa += sqrt((Ey*Hz - Ez*Hy)^2 + (Ez*Hx - Ex*Hz)^2 + (Ex*Hy - Ey*Hx)^2) * dt
    if isgeometry
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
