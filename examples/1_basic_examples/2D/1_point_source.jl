# ******************************************************************************************
# A simple point source.
# ******************************************************************************************
using MaxwellFDTD
using MaxwellPlots

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val

grid = Grid(
    xmin=-50e-6, xmax=50e-6, Nx=501,
    zmin=-50e-6, zmax=50e-6, Nz=501,
)

function source_geometry(x, z)
    return x == 0 && z == 0   # point source at (x,z)=(0,0)
    # return (x == -10e-6 && z == -20e-6) ||   # two point sources at (x,z)=(-10um,-20um)
    #        (x ==  10e-6 && z ==  20e-6)      # and (x,z)=(10um,20um)
    # return sqrt(x^2 + z^2) <= 10e-6   # distributed source in circle of radius 10um
end

function source_waveform(x, z, t)
    lam0 = 2e-6   # (m) central wavelength
    w0 = 2*pi * C0 / lam0   # central frequency
    tau0 = 20e-15   # (s) 1/e pulse duration
    dt0 = 5 * tau0   # (s) time shift needed, since the simulation starts at time t=0
    return exp(-(t - dt0)^2 /tau0^2) * cos(w0 * (t - dt0))
end

source = SoftSource(geometry=source_geometry, waveform=source_waveform, component=:Ex)

model = Model(grid, source, tmax=500e-15, pml=5e-6)

# mybackend = CPU()   # simulate on CPU
mybackend = GPU()   # simulate on GPU
solve!(model, nframes=100, backend=mybackend)

inspect("out.hdf", :Ex, colorrange=(-0.1,0.1))
