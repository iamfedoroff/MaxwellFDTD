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
    inds = geometry2indices(geometry, grid)
    if isempty(inds)
        error("I did not find any grid points which satisfy your monitor geometry.")
    end
    Nt = length(t)
    Hy, Ex = (zeros(Nt) for i=1:2)
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
    inds = geometry2indices(geometry, grid)
    if isempty(inds)
        error("I did not find any grid points which satisfy your monitor geometry.")
    end
    Nt = length(t)
    Hy, Ex, Ez = (zeros(Nt) for i=1:3)
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
    inds = geometry2indices(geometry, grid)
    if isempty(inds)
        error("I did not find any grid points which satisfy your monitor geometry.")
    end
    Nt = length(t)
    Hx, Hy, Hz, Ex, Ey, Ez = (zeros(Nt) for i=1:6)
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
