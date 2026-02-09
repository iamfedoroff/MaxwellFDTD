# ******************************************************************************************
# Interaction with a simple dielectric material.
# ******************************************************************************************
using MaxwellFDTD
using MaxwellPlots

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val

grid = Grid(zmin=-50e-6, zmax=50e-6, Nz=2001)

function source_geometry(z)
    return z == -45e-6   # point source at z=-45um
end

function source_waveform(z, t)
    lam0 = 2e-6   # (m) central wavelength
    w0 = 2*pi * C0 / lam0   # central frequency
    tau0 = 20e-15   # (s) pulse duration
    dt0 = 5 * tau0   # (s) time shift needed, since the simulation starts at time t=0
    return exp(-(t - dt0)^2 /tau0^2) * cos(w0 * (t - dt0))
end

source = SoftSource(geometry=source_geometry, waveform=source_waveform, component=:Ex)

function material_geometry(z)
    return z >= 0   # material occupies half-space z>0
    # return z >= 0 && z <= 20e-6   # material slab from z=0 to z=20um
end

# plot_geometry(grid.z, material_geometry)   # visualize material geometry

glass = Material(geometry=material_geometry, eps=1.5^2)

model = Model(grid, source, tmax=600e-15, pml=5e-6, material=glass)

solve!(model, nframes=100, backend=CPU())

inspect("out.hdf", :Ex)
