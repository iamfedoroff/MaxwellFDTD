mutable struct Output{S, R, A}
    fname :: S
    itout :: Int
    Ntout :: Int
    tout :: R
    Sa :: A   # averaged poynting vector
end


function write_output_variables(out)
    (; fname, Sa) = out
    HDF5.h5open(fname, "r+") do fp
        fp["Sa"] = collect(Sa)
    end
    return nothing
end


# ******************************************************************************************
# 1D: d/dx = d/dy = 0,   (Hy, Ex)
# ******************************************************************************************
function Output(
    model::Model{F}; fname=nothing, nstride=nothing, nframes=nothing, dtout=nothing,
) where F <: Field1D
    (; field, t) = model
    (; grid, Ex) = field
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

    T = eltype(Ex)

    HDF5.h5open(fname, "w") do fp
        fp["z"] = collect(z)
        fp["t"] = collect(tout)
        HDF5.create_dataset(fp, "Hy", T, (Nz, Ntout))
        HDF5.create_dataset(fp, "Ex", T, (Nz, Ntout))
    end

    Sa = zero(Ex)

    return Output(fname, itout, Ntout, tout, Sa)
end


function write_fields(out, model::Model{F}) where F <: Field1D
    (; field) = model
    (; Hy, Ex) = field
    (; fname, itout) = out
    HDF5.h5open(fname, "r+") do fp
        fp["Hy"][:,itout] = collect(Hy)
        fp["Ex"][:,itout] = collect(Ex)
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
) where F <: Field2D
    (; field, t) = model
    (; grid, Ex) = field
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

    T = eltype(Ex)

    HDF5.h5open(fname, "w") do fp
        fp["x"] = collect(x)
        fp["z"] = collect(z)
        fp["t"] = collect(tout)
        HDF5.create_dataset(fp, "Hy", T, (Nx, Nz, Ntout))
        HDF5.create_dataset(fp, "Ex", T, (Nx, Nz, Ntout))
        HDF5.create_dataset(fp, "Ez", T, (Nx, Nz, Ntout))
    end

    Sa = zero(Ex)

    return Output(fname, itout, Ntout, tout, Sa)
end


function write_fields(out, model::Model{F}) where F <: Field2D
    (; field) = model
    (; Hy, Ex, Ez) = field
    (; fname, itout) = out
    HDF5.h5open(fname, "r+") do fp
        fp["Hy"][:,:,itout] = collect(Hy)
        fp["Ex"][:,:,itout] = collect(Ex)
        fp["Ez"][:,:,itout] = collect(Ez)
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
) where F <: Field3D
    (; field, t) = model
    (; grid, Ex) = field
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
    end

    Sa = zero(Ex)

    return Output(fname, itout, Ntout, tout, Sa)
end


function write_fields(out, model::Model{F}) where F <: Field3D
    (; field) = model
    (; Hx, Hy, Hz, Ex, Ey, Ez) = field
    (; fname, itout) = out
    HDF5.h5open(fname, "r+") do fp
        fp["Hx"][:,:,:,itout] = collect(Hx)
        fp["Hy"][:,:,:,itout] = collect(Hy)
        fp["Hz"][:,:,:,itout] = collect(Hz)
        fp["Ex"][:,:,:,itout] = collect(Ex)
        fp["Ey"][:,:,:,itout] = collect(Ey)
        fp["Ez"][:,:,:,itout] = collect(Ez)
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
