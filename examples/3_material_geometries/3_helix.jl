using MaxwellFDTD
using MaxwellPlots


function material_geometry(x, y, z)
    # z1, z2 = 0, 250e-9
    z1, z2 = -250e-9, 0
    R = 60e-9   # (m) helix radius
    r = 35e-9   # (m) radius of helix cross-section
    nt = 1   # number of full twists

    t = 2*pi*nt * (z - z1) / (z2 - z1)
    xshift = R * cos(t)
    yshift = R * sin(t)

    return z >= z1 && z <= z2 && sqrt((x - xshift)^2 + (y - yshift)^2) <= r
end


grid = Grid3D(
    xmin=-200e-9, xmax=200e-9, Nx=201,
    ymin=-200e-9, ymax=200e-9, Ny=201,
    zmin=-400e-9, zmax=400e-9, Nz=401,
)


plot_geometry(grid.x, grid.y, grid.z, material_geometry)
