using MaxwellFDTD
using MaxwellPlots
using RoughEdges


# https://www.comsol.com/blogs/how-to-build-a-parameterized-archimedean-spiral-geometry/
function spiral!(F, x, y; n, width=nothing, phi0=0, theta=0)
    if isnothing(width)
        widths = [0.5, 0.3, 0.2142857, 0.1666666, 0.1363636, 0.1153846, 0.1, 0.0882353, 0.0789474, 0.0714286]
        width = widths[n] * 1e-6
    end

    ai = width   # initial radius
    af = 3e-6   # final radius

    ai1 = ai
    af1 = af
    b1 = (af1 - ai1) / (2*pi*n)
    ai2 = ai + width
    af2 = af + width
    b2 = (af2 - ai2) / (2*pi*n)

    @. F = 0
    s = range(phi0, 2*pi*n, 20001)
    for i in eachindex(s)
        X10 = (ai1 + b1*s[i]) * cos(s[i])
        Y10 = (ai1 + b1*s[i]) * sin(s[i])
        X1 = X10*cos(theta) - Y10*sin(theta)
        Y1 = X10*sin(theta) + Y10*cos(theta)

        X20 = (ai2 + b2*s[i]) * cos(s[i])
        Y20 = (ai2 + b2*s[i]) * sin(s[i])
        X2 = X20*cos(theta) - Y20*sin(theta)
        Y2 = X20*sin(theta) + Y20*cos(theta)

        if X2^2 + Y2^2 <= af2^2
            ix1 = searchsortedfirst(x, X1)
            iy1 = searchsortedfirst(y, Y1)

            ix2 = searchsortedfirst(x, X2)
            iy2 = searchsortedfirst(y, Y2)

            i1, i2 = min(ix1,ix2), max(ix1,ix2)
            j1, j2 = min(iy1,iy2), max(iy1,iy2)
            @. F[i1:i2, j1:j2] = 1
        end
    end

    return nothing
end


grid = Grid3D(
    xmin=-4e-6, xmax=4e-6, Nx=401,
    ymin=-4e-6, ymax=4e-6, Ny=401,
    zmin=-0.4e-6, zmax=0.6e-6, Nz=201,
)


width = 100e-9
S1 = zeros(grid.Nx, grid.Ny)
S2 = zeros(grid.Nx, grid.Ny)
spiral!(S1, grid.x, grid.y; n=5, width, phi0=0, theta=0)
spiral!(S2, grid.x, grid.y; n=5, width, phi0=0, theta=pi)

S = @. S1 + S2

height = 200e-9
@. S = S * height

gmask = geometry_mask(grid.x, grid.y, grid.z, S)

rough_plot(grid.x, grid.y, S; colorrange=(-250,250))
# plot_geometry(grid.x, grid.y, grid.z, gmask)
