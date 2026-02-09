# ******************************************************************************************
# Interaction with multiple dielectric materials.
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

material_geometry_1(z) = z >= 0 && z <= 25e-6   # material slab from z=0 to z=25um
m1 = Material(geometry=material_geometry_1, eps=1.5^2)

material_geometry_2(z) = z >= 5e-6 && z <= 20e-6   # material slab from z=5um to z=20um
m2 = Material(geometry=material_geometry_2, eps=2^2)

material_geometry_3(z) = z >= 10e-6 && z <= 15e-6   # material slab from z=10um to z=15um
m3 = Material(geometry=material_geometry_3, eps=2.5^2)

model = Model(grid, source, tmax=1200e-15, pml=5e-6, material=(m1,m2,m3))

# plot_geometry(grid.z, model.geometry)   # visualize material geometry

solve!(model, nframes=100, backend=CPU())

inspect("out.hdf", :Ex)
