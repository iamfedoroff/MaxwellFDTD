abstract type Monitor end


"""
FieldMonitor{G} <: Monitor

Field monitor accumulates fields in a given area of space

# Fields
- `geometry::G<:Function`: Function of grid coordinates (or the corresponding array) which
    defines an area of space in which the monitor accumulates the components of the field:
    the area consists of spatial points for which the value of geometry is true.
"""
struct FieldMonitor{G} <: Monitor
    geometry :: G
end


# ******************************************************************************************
struct FieldMonitor1D{C, A}
    inds :: C
    Hy :: A
    Ex :: A
end


function Monitor(monitor::FieldMonitor, grid::Grid1D, t)
    (; geometry) = monitor
    (; Nz, z) = grid
    Nt = length(t)
    if geometry isa Function
        inds = [CartesianIndex(iz) for iz=1:Nz if geometry(z[iz])]
    else
        inds = [CartesianIndex(iz) for iz=1:Nz if Bool(geometry[iz])]
    end
    Hy = zeros(Nt)
    Ex = zeros(Nt)
    return FieldMonitor1D(inds, Hy, Ex)
end


function update_monitor!(monitor::FieldMonitor1D, field, it)
    (; inds, Hy, Ex) = monitor
    Hy[it] = @views sum(field.Hy[inds])
    Ex[it] = @views sum(field.Ex[inds])
    return nothing
end


function write_monitor(fp, n, monitor::FieldMonitor1D)
    fp["monitors/$n/inds"] = monitor.inds
    fp["monitors/$n/Hy"] = monitor.Hy
    fp["monitors/$n/Ex"] = monitor.Ex
    return nothing
end


# ******************************************************************************************
struct FieldMonitor2D{C, A}
    inds :: C
    Hy :: A
    Ex :: A
    Ez :: A
end


function Monitor(monitor::FieldMonitor, grid::Grid2D, t)
    (; geometry) = monitor
    (; Nx, Nz, x, z) = grid
    Nt = length(t)
    if geometry isa Function
        inds = [CartesianIndex(ix,iz) for ix=1:Nx, iz=1:Nz if geometry(x[ix],z[iz])]
    else
        inds = [CartesianIndex(ix,iz) for ix=1:Nx, iz=1:Nz if Bool(geometry[ix,iz])]
    end
    Hy = zeros(Nt)
    Ex, Ez = zeros(Nt), zeros(Nt)
    return FieldMonitor2D(inds, Hy, Ex, Ez)
end


function update_monitor!(monitor::FieldMonitor2D, field, it)
    (; inds, Hy, Ex, Ez) = monitor
    Hy[it] = @views sum(field.Hy[inds])
    Ex[it] = @views sum(field.Ex[inds])
    Ez[it] = @views sum(field.Ez[inds])
    return nothing
end


function write_monitor(fp, n, monitor::FieldMonitor2D)
    fp["monitors/$n/inds"] = monitor.inds
    fp["monitors/$n/Hy"] = monitor.Hy
    fp["monitors/$n/Ex"] = monitor.Ex
    fp["monitors/$n/Ez"] = monitor.Ez
    return nothing
end


# ******************************************************************************************
struct FieldMonitor3D{C, A}
    inds :: C
    Hx :: A
    Hy :: A
    Hz :: A
    Ex :: A
    Ey :: A
    Ez :: A
end


function Monitor(monitor::FieldMonitor, grid::Grid3D, t)
    (; geometry) = monitor
    (; Nx, Ny, Nz, x, y, z) = grid
    Nt = length(t)
    if geometry isa Function
        inds = [
            CartesianIndex(ix,iy,iz)
            for ix=1:Nx, iy=1:Ny, iz=1:Nz if geometry(x[ix],y[iy],z[iz])
        ]
    else
        inds = [
            CartesianIndex(ix,iy,iz)
            for ix=1:Nx, iy=1:Ny, iz=1:Nz if Bool(geometry[ix,iy,iz])
        ]
    end
    Hx, Hy, Hz = (zeros(Nt) for i=1:3)
    Ex, Ey, Ez = (zeros(Nt) for i=1:3)
    return FieldMonitor3D(inds, Hx, Hy, Hz, Ex, Ey, Ez)
end


function update_monitor!(monitor::FieldMonitor3D, field, it)
    (; inds, Hx, Hy, Hz, Ex, Ey, Ez) = monitor
    Hx[it] = @views sum(field.Hx[inds])
    Hy[it] = @views sum(field.Hy[inds])
    Hz[it] = @views sum(field.Hz[inds])
    Ex[it] = @views sum(field.Ex[inds])
    Ey[it] = @views sum(field.Ey[inds])
    Ez[it] = @views sum(field.Ez[inds])
    return nothing
end


function write_monitor(fp, n, monitor::FieldMonitor3D)
    fp["monitors/$n/inds"] = monitor.inds
    fp["monitors/$n/Hx"] = monitor.Hx
    fp["monitors/$n/Hy"] = monitor.Hy
    fp["monitors/$n/Hz"] = monitor.Hz
    fp["monitors/$n/Ex"] = monitor.Ex
    fp["monitors/$n/Ey"] = monitor.Ey
    fp["monitors/$n/Ez"] = monitor.Ez
    return nothing
end
