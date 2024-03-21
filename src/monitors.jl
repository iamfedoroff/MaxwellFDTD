abstract type Monitor end


# ******************************************************************************************
# Field monitors
# ******************************************************************************************
struct FieldMonitor{G} <: Monitor
    geometry :: G
    issum :: Bool
end


"""
FieldMonitor(geometry; sum=false)

Field monitor records the fields in time for a given area of space.

# Arguments
- `geometry::Union{Function,AbstractArray}`: Function of grid coordinates (or the
    corresponding array) which defines an area of space in which the monitor records the
    components of the field: the area consists of spatial points for which the value of
    geometry is true.

# Keywords
- `sum::Bool=false`: If false, then record the field in each spatial point independently. If
    true, then sum the fields from different spatial points.
"""
function FieldMonitor(geometry; sum=false)
    return FieldMonitor(geometry, sum)
end



# ******************************************************************************************
struct FieldMonitor1D{C, A}
    issum :: Bool
    inds :: C
    Hy :: A
    Ex :: A
end


function Monitor(monitor::FieldMonitor, grid::Grid1D, t)
    (; geometry, issum) = monitor
    inds = geometry2indices(geometry, grid)
    if isempty(inds)
        error("I did not find any grid points which satisfy your monitor geometry.")
    end
    Nt = length(t)
    if issum
        Hy, Ex = (zeros(Nt) for i=1:2)
    else
        Hy, Ex = (zeros(length(inds), Nt) for i=1:2)
    end
    return FieldMonitor1D(issum, inds, Hy, Ex)
end


function update_monitor!(monitor::FieldMonitor1D, field, it)
    (; issum, inds, Hy, Ex) = monitor
    if issum
        Hy[it] = @views sum(field.Hy[inds])
        Ex[it] = @views sum(field.Ex[inds])
    else
        Hy[:,it] .= collect(field.Hy[inds])
        Ex[:,it] .= collect(field.Ex[inds])
    end
    return nothing
end


function write_monitor(fp, n, monitor::FieldMonitor1D)
    fp["monitors/$n/issum"] = monitor.issum
    fp["monitors/$n/inds"] = monitor.inds
    fp["monitors/$n/Hy"] = monitor.Hy
    fp["monitors/$n/Ex"] = monitor.Ex
    return nothing
end


# ******************************************************************************************
struct FieldMonitor2D{C, A}
    issum :: Bool
    inds :: C
    Hy :: A
    Ex :: A
    Ez :: A
end


function Monitor(monitor::FieldMonitor, grid::Grid2D, t)
    (; geometry, issum) = monitor
    inds = geometry2indices(geometry, grid)
    if isempty(inds)
        error("I did not find any grid points which satisfy your monitor geometry.")
    end
    Nt = length(t)
    if issum
        Hy, Ex, Ez = (zeros(Nt) for i=1:3)
    else
        Hy, Ex, Ez = (zeros(length(inds), Nt) for i=1:3)
    end
    return FieldMonitor2D(issum, inds, Hy, Ex, Ez)
end


function update_monitor!(monitor::FieldMonitor2D, field, it)
    (; issum, inds, Hy, Ex, Ez) = monitor
    if issum
        Hy[it] = @views sum(field.Hy[inds])
        Ex[it] = @views sum(field.Ex[inds])
        Ez[it] = @views sum(field.Ez[inds])
    else
        Hy[:,it] .= collect(field.Hy[inds])
        Ex[:,it] .= collect(field.Ex[inds])
        Ez[:,it] .= collect(field.Ez[inds])
    end
    return nothing
end


function write_monitor(fp, n, monitor::FieldMonitor2D)
    fp["monitors/$n/issum"] = monitor.issum
    fp["monitors/$n/inds"] = monitor.inds
    fp["monitors/$n/Hy"] = monitor.Hy
    fp["monitors/$n/Ex"] = monitor.Ex
    fp["monitors/$n/Ez"] = monitor.Ez
    return nothing
end


# ******************************************************************************************
struct FieldMonitor3D{C, A}
    issum :: Bool
    inds :: C
    Hx :: A
    Hy :: A
    Hz :: A
    Ex :: A
    Ey :: A
    Ez :: A
end


function Monitor(monitor::FieldMonitor, grid::Grid3D, t)
    (; geometry, issum) = monitor
    inds = geometry2indices(geometry, grid)
    if isempty(inds)
        error("I did not find any grid points which satisfy your monitor geometry.")
    end
    Nt = length(t)
    if issum
        Hx, Hy, Hz, Ex, Ey, Ez = (zeros(Nt) for i=1:6)
    else
        Hx, Hy, Hz, Ex, Ey, Ez = (zeros(length(inds), Nt) for i=1:6)
    end
    return FieldMonitor3D(issum, inds, Hx, Hy, Hz, Ex, Ey, Ez)
end


function update_monitor!(monitor::FieldMonitor3D, field, it)
    (; issum, inds, Hx, Hy, Hz, Ex, Ey, Ez) = monitor
    if issum
        Hx[it] = @views sum(field.Hx[inds])
        Hy[it] = @views sum(field.Hy[inds])
        Hz[it] = @views sum(field.Hz[inds])
        Ex[it] = @views sum(field.Ex[inds])
        Ey[it] = @views sum(field.Ey[inds])
        Ez[it] = @views sum(field.Ez[inds])
    else
        Hx[:,it] .= collect(field.Hx[inds])
        Hy[:,it] .= collect(field.Hy[inds])
        Hz[:,it] .= collect(field.Hz[inds])
        Ex[:,it] .= collect(field.Ex[inds])
        Ey[:,it] .= collect(field.Ey[inds])
        Ez[:,it] .= collect(field.Ez[inds])
    end
    return nothing
end


function write_monitor(fp, n, monitor::FieldMonitor3D)
    fp["monitors/$n/issum"] = monitor.issum
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
    issum :: Bool
end


"""
SpectralMonitor(geometry; sum=false)

Spectral monitor records the spectrum of fields in a given area of space.

# Arguments
- `geometry::Union{Function,AbstractArray}`: Function of grid coordinates (or the
    corresponding array) which defines an area of space in which the monitor records the
    spectra of the field components: the area consists of spatial points for which the value
    of geometry is true.

# Keywords
- `sum::Bool=false`: If false, then record the spectra in each spatial point independently.
    If true, then sum the spectra from different spatial points.
"""
function SpectralMonitor(geometry; wmin=nothing, wmax=nothing, sum=false)
    return SpectralMonitor(geometry, wmin, wmax, sum)
end


# ******************************************************************************************
struct SpectralMonitor1D{C, D, T, A}
    issum :: Bool
    inds :: C
    dft :: D
    t :: T
    Hy :: A
    Ex :: A
end


function Monitor(monitor::SpectralMonitor, grid::Grid1D, t)
    (; geometry, wmin, wmax, issum) = monitor
    inds = geometry2indices(geometry, grid)
    if isempty(inds)
        error("I did not find any grid points which satisfy your monitor geometry.")
    end
    dft = DFT(t; wmin, wmax, sum=issum)
    if issum
        Hy, Ex = (zeros(ComplexF64, dft.Nw) for i=1:2)
    else
        Hy, Ex = (zeros(ComplexF64, length(inds), dft.Nw) for i=1:2)
    end
    return SpectralMonitor1D(issum, inds, dft, t, Hy, Ex)
end


function update_monitor!(monitor::SpectralMonitor1D, field, it)
    (; inds, dft, t, Hy, Ex) = monitor
    dft(Hy, collect(field.Hy[inds]), t[it])
    dft(Ex, collect(field.Ex[inds]), t[it])
    return nothing
end


function write_monitor(fp, n, monitor::SpectralMonitor1D)
    fp["monitors/$n/issum"] = monitor.issum
    fp["monitors/$n/inds"] = monitor.inds
    fp["monitors/$n/w"] = monitor.dft.w
    fp["monitors/$n/Hy"] = monitor.Hy
    fp["monitors/$n/Ex"] = monitor.Ex
    return nothing
end


# ******************************************************************************************
struct SpectralMonitor2D{C, D, T, A}
    issum :: Bool
    inds :: C
    dft :: D
    t :: T
    Hy :: A
    Ex :: A
    Ez :: A
end


function Monitor(monitor::SpectralMonitor, grid::Grid2D, t)
    (; geometry, wmin, wmax, issum) = monitor
    inds = geometry2indices(geometry, grid)
    if isempty(inds)
        error("I did not find any grid points which satisfy your monitor geometry.")
    end
    dft = DFT(t; wmin, wmax, sum=issum)
    if issum
        Hy, Ex, Ez = (zeros(ComplexF64, dft.Nw) for i=1:3)
    else
        Hy, Ex, Ez = (zeros(ComplexF64, length(inds), dft.Nw) for i=1:3)
    end
    return SpectralMonitor2D(issum, inds, dft, t, Hy, Ex, Ez)
end


function update_monitor!(monitor::SpectralMonitor2D, field, it)
    (; inds, dft, t, Hy, Ex, Ez) = monitor
    dft(Hy, collect(field.Hy[inds]), t[it])
    dft(Ex, collect(field.Ex[inds]), t[it])
    dft(Ez, collect(field.Ez[inds]), t[it])
    return nothing
end


function write_monitor(fp, n, monitor::SpectralMonitor2D)
    fp["monitors/$n/issum"] = monitor.issum
    fp["monitors/$n/inds"] = monitor.inds
    fp["monitors/$n/w"] = monitor.dft.w
    fp["monitors/$n/Hy"] = monitor.Hy
    fp["monitors/$n/Ex"] = monitor.Ex
    fp["monitors/$n/Ez"] = monitor.Ez
    return nothing
end


# ******************************************************************************************
struct SpectralMonitor3D{C, D, T, A}
    issum :: Bool
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
    (; geometry, wmin, wmax, issum) = monitor
    inds = geometry2indices(geometry, grid)
    if isempty(inds)
        error("I did not find any grid points which satisfy your monitor geometry.")
    end
    dft = DFT(t; wmin, wmax, sum=issum)
    if issum
        Hx, Hy, Hz, Ex, Ey, Ez = (zeros(ComplexF64, dft.Nw) for i=1:6)
    else
        Hx, Hy, Hz, Ex, Ey, Ez = (zeros(ComplexF64, length(inds), dft.Nw) for i=1:6)
    end
    return SpectralMonitor3D(issum, inds, dft, t, Hx, Hy, Hz, Ex, Ey, Ez)
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
    fp["monitors/$n/issum"] = monitor.issum
    fp["monitors/$n/inds"] = monitor.inds
    fp["monitors/$n/w"] = monitor.dft.w
    fp["monitors/$n/Hx"] = monitor.Hx
    fp["monitors/$n/Hy"] = monitor.Hy
    fp["monitors/$n/Hz"] = monitor.Hz
    fp["monitors/$n/Ex"] = monitor.Ex
    fp["monitors/$n/Ey"] = monitor.Ey
    fp["monitors/$n/Ez"] = monitor.Ez
    return nothing
end
