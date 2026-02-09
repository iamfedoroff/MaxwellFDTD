using MaxwellFDTD
using MaxwellPlots


function material_geometry(x, y, z)
    theta = deg2rad(45)
    h = 5e-6
    return (x^2 + y^2) * cos(theta)^2 - z^2 * sin(theta)^2 <= 0 && z >= 0 && z <= h
end


grid = Grid3D(
    xmin=-10e-6, xmax=10e-6, Nx=201,
    ymin=-10e-6, ymax=10e-6, Ny=201,
    zmin=-10e-6, zmax=10e-6, Nz=201,
)


plot_geometry(grid.x, grid.y, grid.z, material_geometry)
