# ******************************************************************************************
# A simple point source.
# ******************************************************************************************
using MaxwellFDTD
using MaxwellPlots

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val

grid = Grid(
    xmin=-25e-6, xmax=25e-6, Nx=201,
    ymin=-25e-6, ymax=25e-6, Ny=201,
    zmin=-25e-6, zmax=25e-6, Nz=201,
)

function source_geometry(x, y, z)
    return x == 0 && y == 0 && z == 0   # point source at (x,y,z)=(0,0,0)
    # return (x == -5e-6 && y ==  5e-6 && z == -10e-6) ||   # two point sources
    #        (x ==  5e-6 && y == -5e-6 && z ==  10e-6)
    # return sqrt(x^2 + y^2 + z^2) <= 5e-6   # distributed source in sphere of radius 5um
end

function source_waveform(x, y, z, t)
    lam0 = 2e-6   # (m) central wavelength
    w0 = 2*pi * C0 / lam0   # central frequency
    tau0 = 20e-15   # (s) 1/e pulse duration
    dt0 = 5 * tau0   # (s) time shift needed, since the simulation starts at time t=0
    return exp(-(t - dt0)^2 /tau0^2) * cos(w0 * (t - dt0))
end

source = SoftSource(geometry=source_geometry, waveform=source_waveform, component=:Ex)

model = Model(grid, source, tmax=300e-15, pml=2e-6)

# mybackend = CPU()   # simulate on CPU
mybackend = GPU()   # simulate on GPU
solve!(model, nframes=50, backend=mybackend)

inspect("out.hdf", :Ex, colorrange=(-0.01,0.01))
