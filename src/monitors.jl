abstract type Monitor end


# ******************************************************************************************
# Field monitors
# ******************************************************************************************
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
    Hy, Ex = (zeros(length(inds), Nt) for i=1:2)
    return FieldMonitor1D(inds, Hy, Ex)
end


function update_monitor!(monitor::FieldMonitor1D, field, it)
    (; inds, Hy, Ex) = monitor
    Hy[:,it] .= collect(field.Hy[inds])
    Ex[:,it] .= collect(field.Ex[inds])
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
    Hy, Ex, Ez = (zeros(length(inds), Nt) for i=1:3)
    return FieldMonitor2D(inds, Hy, Ex, Ez)
end


function update_monitor!(monitor::FieldMonitor2D, field, it)
    (; inds, Hy, Ex, Ez) = monitor
    Hy[:,it] .= collect(field.Hy[inds])
    Ex[:,it] .= collect(field.Ex[inds])
    Ez[:,it] .= collect(field.Ez[inds])
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
    Hx, Hy, Hz, Ex, Ey, Ez = (zeros(length(inds), Nt) for i=1:6)
    return FieldMonitor3D(inds, Hx, Hy, Hz, Ex, Ey, Ez)
end


function update_monitor!(monitor::FieldMonitor3D, field, it)
    (; inds, Hx, Hy, Hz, Ex, Ey, Ez) = monitor
    Hx[:,it] .= collect(field.Hx[inds])
    Hy[:,it] .= collect(field.Hy[inds])
    Hz[:,it] .= collect(field.Hz[inds])
    Ex[:,it] .= collect(field.Ex[inds])
    Ey[:,it] .= collect(field.Ey[inds])
    Ez[:,it] .= collect(field.Ez[inds])
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


# ******************************************************************************************
# Spectral Monitors
# ******************************************************************************************
struct SpectralMonitor{G, T1, T2} <: Monitor
    geometry :: G
    wmin :: T1
    wmax :: T2
end


function SpectralMonitor(geometry; wmin=nothing, wmax=nothing)
    return SpectralMonitor(geometry, wmin, wmax)
end


# ******************************************************************************************
struct SpectralMonitor1D{C, D, T, A}
    inds :: C
    dft :: D
    t :: T
    Hy :: A
    Ex :: A
end


function Monitor(monitor::SpectralMonitor, grid::Grid1D, t)
    (; geometry, wmin, wmax) = monitor
    inds = geometry2indices(geometry, grid)
    if isempty(inds)
        error("I did not find any grid points which satisfy your monitor geometry.")
    end
    dft = DFT(t; wmin, wmax)
    Hy, Ex = (zeros(ComplexF64, dft.Nw) for i=1:2)
    return SpectralMonitor1D(inds, dft, t, Hy, Ex)
end


function update_monitor!(monitor::SpectralMonitor1D, field, it)
    (; inds, dft, t, Hy, Ex) = monitor
    dft(Hy, collect(field.Hy[inds]), t[it])
    dft(Ex, collect(field.Ex[inds]), t[it])
    return nothing
end


function write_monitor(fp, n, monitor::SpectralMonitor1D)
    (; dft) = monitor
    fp["monitors/$n/inds"] = monitor.inds
    fp["monitors/$n/w"] = dft.w
    fp["monitors/$n/Hy"] = 2 * monitor.Hy * dft.dt
    fp["monitors/$n/Ex"] = 2 * monitor.Ex * dft.dt
    return nothing
end


# ******************************************************************************************
struct SpectralMonitor2D{C, D, T, A}
    inds :: C
    dft :: D
    t :: T
    Hy :: A
    Ex :: A
    Ez :: A
end


function Monitor(monitor::SpectralMonitor, grid::Grid2D, t)
    (; geometry, wmin, wmax) = monitor
    inds = geometry2indices(geometry, grid)
    if isempty(inds)
        error("I did not find any grid points which satisfy your monitor geometry.")
    end
    dft = DFT(t; wmin, wmax)
    Hy, Ex, Ez = (zeros(ComplexF64, dft.Nw) for i=1:3)
    return SpectralMonitor2D(inds, dft, t, Hy, Ex, Ez)
end


function update_monitor!(monitor::SpectralMonitor2D, field, it)
    (; inds, dft, t, Hy, Ex, Ez) = monitor
    dft(Hy, collect(field.Hy[inds]), t[it])
    dft(Ex, collect(field.Ex[inds]), t[it])
    dft(Ez, collect(field.Ez[inds]), t[it])
    return nothing
end


function write_monitor(fp, n, monitor::SpectralMonitor2D)
    (; dft) = monitor
    fp["monitors/$n/inds"] = monitor.inds
    fp["monitors/$n/w"] = dft.w
    fp["monitors/$n/Hy"] = 2 * monitor.Hy * dft.dt
    fp["monitors/$n/Ex"] = 2 * monitor.Ex * dft.dt
    fp["monitors/$n/Ez"] = 2 * monitor.Ez * dft.dt
    return nothing
end


# ******************************************************************************************
struct SpectralMonitor3D{C, D, T, A}
    inds :: C
    dft :: D
    t :: T
    Hx :: A
    Hy :: A
    Hz :: A
    Ex :: A
    Ey :: A
    Ez :: A
end


function Monitor(monitor::SpectralMonitor, grid::Grid3D, t)
    (; geometry, wmin, wmax) = monitor
    inds = geometry2indices(geometry, grid)
    if isempty(inds)
        error("I did not find any grid points which satisfy your monitor geometry.")
    end
    dft = DFT(t; wmin, wmax)
    Hx, Hy, Hz, Ex, Ey, Ez = (zeros(ComplexF64, dft.Nw) for i=1:6)
    return SpectralMonitor3D(inds, dft, t, Hx, Hy, Hz, Ex, Ey, Ez)
end


function update_monitor!(monitor::SpectralMonitor3D, field, it)
    (; inds, dft, t, Hx, Hy, Hz, Ex, Ey, Ez) = monitor
    dft(Hx, collect(field.Hx[inds]), t[it])
    dft(Hy, collect(field.Hy[inds]), t[it])
    dft(Hz, collect(field.Hz[inds]), t[it])
    dft(Ex, collect(field.Ex[inds]), t[it])
    dft(Ey, collect(field.Ey[inds]), t[it])
    dft(Ez, collect(field.Ez[inds]), t[it])
    return nothing
end


function write_monitor(fp, n, monitor::SpectralMonitor3D)
    (; dft) = monitor
    fp["monitors/$n/inds"] = monitor.inds
    fp["monitors/$n/w"] = dft.w
    fp["monitors/$n/Hx"] = 2 * monitor.Hx * dft.dt
    fp["monitors/$n/Hy"] = 2 * monitor.Hy * dft.dt
    fp["monitors/$n/Hz"] = 2 * monitor.Hz * dft.dt
    fp["monitors/$n/Ex"] = 2 * monitor.Ex * dft.dt
    fp["monitors/$n/Ey"] = 2 * monitor.Ey * dft.dt
    fp["monitors/$n/Ez"] = 2 * monitor.Ez * dft.dt
    return nothing
end
