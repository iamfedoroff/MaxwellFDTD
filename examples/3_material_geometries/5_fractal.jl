using MaxwellFDTD
using MaxwellPlots
using RoughEdges

import KernelAbstractions: @index, @kernel, get_backend


@kernel function julia_set_kernel!(J, x, y, c, Nz, zmax, scale)
    ix, iy = @index(Global, NTuple)
    @inbounds begin
        iz = 1
        z = (x[ix] + 1im * y[iy]) / scale
        while abs(z) <= zmax && iz < Nz
            z = z*z + c
            iz += 1
        end
        J[ix,iy] = (iz-1) / (Nz-1)
    end
end
function julia_set!(J, x, y; c, Nz=1000, zmax=10, scale=1)
    backend = get_backend(J)
    ndrange = size(J)
    julia_set_kernel!(backend)(J, x, y, c, Nz, zmax, scale; ndrange)
end


"""
https://julialang.org/blog/2016/02/iteration/#a_multidimensional_boxcar_filter
"""
function moving_average(A::AbstractArray, m::Int)
    if eltype(A) == Int
        out = zeros(size(A))
    else
        out = similar(A)
    end
    R = CartesianIndices(A)
    Ifirst, Ilast = first(R), last(R)
    I1 = div(m,2) * oneunit(Ifirst)
    for I in R
        n, s = 0, zero(eltype(out))
        for J in max(Ifirst, I-I1):min(Ilast, I+I1)
            s += A[J]
            n += 1
        end
        out[I] = s/n
    end
    return out
end


# ******************************************************************************************
grid = Grid3D(
    xmin=-2e-6, xmax=2e-6, Nx=1001,
    ymin=-2e-6, ymax=2e-6, Ny=1001,
    zmin=-0.4e-6, zmax=0.6e-6, Nz=201,
)


# https://paulbourke.net/fractals/juliaset/
# c = 0.355 + 0.355im
# c = -0.4 - 0.59im

# https://www.karlsims.com/julia.html
# c = 0.3 - 0.01im

# https://en.wikipedia.org/wiki/File:Julia_set,_plotted_with_Matplotlib.svg
c = -0.5125 + 0.5213im

J = zeros(grid.Nx, grid.Ny)
julia_set!(J, grid.x, grid.y; c, scale=1e-6)
J .= J .- minimum(J)
J .= J ./ maximum(J)

# saturate:
sat = 0.3
for i in eachindex(J)
    if J[i] >= sat
        J[i] = sat
    end
end
J .= J .- minimum(J)
J .= J ./ maximum(J)


# smooth:
J = moving_average(J, 3)
J .= J .- minimum(J)
J .= J ./ maximum(J)

height = -100e-9
@. J = J * height


gmask = geometry_mask(grid.x, grid.y, grid.z, J)


# ******************************************************************************************
rough_plot(grid.x, grid.y, J)
# plot_geometry(grid.x, grid.y, grid.z, gmask)
