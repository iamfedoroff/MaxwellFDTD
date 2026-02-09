using MaxwellFDTD
using MaxwellPlots


function material_geometry(x, y, z)
    G = z <= 0   # half space
    # G = z >= -1e-6 && z <= 0   # slab
    # G = z >= -1e-6 && z <= 0 && !(abs(x) <= 0.5e-6)   # slit
    # G = z >= -1e-6 && z <= 0 &&
    #     !(abs(x-2e-6) <= 0.5e-6 || abs(x+2e-6) <= 0.5e-6)   # double slit
    # G = z >= -1e-6 && z <= 0 && !(sqrt(x^2 + y^2) <= 2e-6)   # hole
    # G = z >= -1e-6 && z <= 0 && !(sqrt((x-3e-6)^2 + y^2) <= 2e-6 ||
    #     sqrt((x+3e-6)^2 + y^2) <= 2e-6)   # two holes
    # G = sqrt(x^2 + y^2 + z^2) <= 3e-6   # ball
    # G = sqrt((x-3e-6)^2 + (y-3e-6)^2 + (z-3e-6)^2) <= 3e-6 ||
    #     sqrt((x+3e-6)^2 + (y+3e-6)^2 + (z+3e-6)^2) <= 3e-6  # two balls
    # G = z <= 0.5*x - 1e-6   # inclined interface
    # G = z <= 1e-6 * cos(2*pi/5e-6 * x)   # wavy surface
    # G = z <= 1e-6 * cos(2*pi/5e-6 * x) * cos(2*pi/5e-6 * y)   # egg tray surface

    # merge two shapes:
    # G1 = sqrt(x^2 + y^2 + z^2) <= 3e-6   # ball
    # G2 = z <= 0   # plane interface
    # G = G1 || G2

    # subtract two shapes:
    # G1 = z <= 0   # plane interface
    # G2 = sqrt(x^2 + y^2 + z^2) <= 3e-6   # ball
    # G = G1 && !G2

    return G
end


grid = Grid3D(
    xmin=-10e-6, xmax=10e-6, Nx=201,
    ymin=-10e-6, ymax=10e-6, Ny=201,
    zmin=-10e-6, zmax=10e-6, Nz=201,
)


plot_geometry(grid.x, grid.y, grid.z, material_geometry)
