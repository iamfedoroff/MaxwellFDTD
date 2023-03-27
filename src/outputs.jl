mutable struct Output{S, R}
    fname :: S
    itout :: Int
    Ntout :: Int
    tout :: R
end


function Output(
    model;
    fname="results/1d_out.hdf", nstride=nothing, nframes=nothing, dtout=nothing,
)
    (; field, Nt, t) = model
    (; grid) = field

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
    tout = t[1:nstride:end]
    Ntout = length(tout)
    itout = 1

    if !isdir(dirname(fname))
        mkpath(dirname(fname))
    end

    HDF5.h5open(fname, "w") do fp
        fp["t"] = collect(tout)
        prepare_output!(fp, Ntout, grid)
    end

    return Output(fname, itout, Ntout, tout)
end


# ******************************************************************************
# 1D: d/dx = d/dy = 0,   (Hy, Ex)
# ******************************************************************************
function prepare_output!(fp, Ntout, grid::Grid1D{T}) where T
    (; Nz, z) = grid
    fp["z"] = collect(z)
    HDF5.create_dataset(fp, "Hy", T, (Nz, Ntout))
    HDF5.create_dataset(fp, "Ex", T, (Nz, Ntout))
    return nothing
end


function write_output!(out, model::Model1D)
    (; field) = model
    (; Hy, Ex) = field
    (; fname, itout) = out
    HDF5.h5open(fname, "r+") do fp
        fp["Hy"][:,itout] = collect(Hy)
        fp["Ex"][:,itout] = collect(Ex)
    end
    return nothing
end


# ******************************************************************************
# 2D: d/dz = 0,   (Hx, Hy, Ez)
# ******************************************************************************
function prepare_output!(fp, Ntout, grid::Grid2D{T}) where T
    (; Nx, Ny, x, y) = grid
    fp["x"] = collect(x)
    fp["y"] = collect(y)
    HDF5.create_dataset(fp, "Hx", T, (Nx, Ny, Ntout))
    HDF5.create_dataset(fp, "Hy", T, (Nx, Ny, Ntout))
    HDF5.create_dataset(fp, "Ez", T, (Nx, Ny, Ntout))
    return nothing
end


function write_output!(out, model::Model2D)
    (; field) = model
    (; Hx, Hy, Ez) = field
    (; fname, itout) = out
    HDF5.h5open(fname, "r+") do fp
        fp["Hx"][:,:,itout] = collect(Hx)
        fp["Hy"][:,:,itout] = collect(Hy)
        fp["Ez"][:,:,itout] = collect(Ez)
    end
    return nothing
end


# ******************************************************************************
# 3D
# ******************************************************************************
function prepare_output!(fp, Ntout, grid::Grid3D{T}) where T
    (; Nx, Ny, Nz, x, y, z) = grid
    fp["x"] = collect(x)
    fp["y"] = collect(y)
    fp["z"] = collect(z)
    HDF5.create_dataset(fp, "Hx", T, (Nx, Ny, Nz, Ntout))
    HDF5.create_dataset(fp, "Hy", T, (Nx, Ny, Nz, Ntout))
    HDF5.create_dataset(fp, "Hz", T, (Nx, Ny, Nz, Ntout))
    HDF5.create_dataset(fp, "Ex", T, (Nx, Ny, Nz, Ntout))
    HDF5.create_dataset(fp, "Ey", T, (Nx, Ny, Nz, Ntout))
    HDF5.create_dataset(fp, "Ez", T, (Nx, Ny, Nz, Ntout))
    return nothing
end


function write_output!(out, model::Model3D)
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
