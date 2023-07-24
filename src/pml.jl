# ******************************************************************************************
# 1D
# ******************************************************************************************
struct PML1D{A}
    iz1 :: Int
    iz2 :: Int
    Kz :: A
    Az :: A
    Bz :: A
    psiExz :: A
    psiHyz :: A
end

@adapt_structure PML1D


function PML(z, box, dt; kappa=1, alpha=10e-6, R0=10e-6, m=3)
    Nz = length(z)
    Kz, Az, Bz = pml_coefficients(z, box, dt; kappa, alpha, R0, m)
    psiExz, psiHyz = zeros(Nz), zeros(Nz)

    Lz1, Lz2 = box
    z1, z2 = z[1] + Lz1, z[end] - Lz2
    iz1, iz2 = argmin(abs.(z .- z1)), argmin(abs.(z .- z2))

    return PML1D(iz1, iz2, Kz, Az, Bz, psiExz, psiHyz)
end


# ******************************************************************************************
# 2D
# ******************************************************************************************
struct PML2D{V, A}
    ix1 :: Int
    ix2 :: Int
    iz1 :: Int
    iz2 :: Int
    Kx :: V
    Ax :: V
    Bx :: V
    Kz :: V
    Az :: V
    Bz :: V
    psiExz :: A
    psiEzx :: A
    psiHyx :: A
    psiHyz :: A
end

@adapt_structure PML2D


function PML(x, z, box, dt)
    Nx, Nz = length(x), length(z)
    Kx, Ax, Bx = pml_coefficients(x, box[1:2], dt)
    Kz, Az, Bz = pml_coefficients(z, box[3:4], dt)
    psiExz, psiEzx, psiHyx, psiHyz = (zeros(Nx,Nz) for i=1:4)

    Lx1, Lx2, Lz1, Lz2 = box
    x1, x2 = x[1] + Lx1, x[end] - Lx2
    z1, z2 = z[1] + Lz1, z[end] - Lz2
    ix1, ix2 = argmin(abs.(x .- x1)), argmin(abs.(x .- x2))
    iz1, iz2 = argmin(abs.(z .- z1)), argmin(abs.(z .- z2))

    return PML2D(ix1, ix2, iz1, iz2, Kx, Ax, Bx, Kz, Az, Bz, psiExz, psiEzx, psiHyx, psiHyz)
end


# ******************************************************************************************
# 3D
# ******************************************************************************************
struct PML3D{V, A}
    ix1 :: Int
    ix2 :: Int
    iy1 :: Int
    iy2 :: Int
    iz1 :: Int
    iz2 :: Int
    Kx :: V
    Ax :: V
    Bx :: V
    Ky :: V
    Ay :: V
    By :: V
    Kz :: V
    Az :: V
    Bz :: V
    psiExy :: A
    psiExz :: A
    psiEyx :: A
    psiEyz :: A
    psiEzx :: A
    psiEzy :: A
    psiHxy :: A
    psiHxz :: A
    psiHyx :: A
    psiHyz :: A
    psiHzx :: A
    psiHzy :: A
end

@adapt_structure PML3D


function PML(x, y, z, box, dt)
    Nx, Ny, Nz = length(x), length(y), length(z)
    Kx, Ax, Bx = pml_coefficients(x, box[1:2], dt)
    Ky, Ay, By = pml_coefficients(y, box[3:4], dt)
    Kz, Az, Bz = pml_coefficients(z, box[5:6], dt)
    psiExy, psiExz, psiEyx, psiEyz, psiEzx, psiEzy = (zeros(Nx,Ny,Nz) for i=1:6)
    psiHxy, psiHxz, psiHyx, psiHyz, psiHzx, psiHzy = (zeros(Nx,Ny,Nz) for i=1:6)

    Lx1, Lx2, Ly1, Ly2, Lz1, Lz2 = box
    x1, x2 = x[1] + Lx1, x[end] - Lx2
    y1, y2 = y[1] + Ly1, y[end] - Ly2
    z1, z2 = z[1] + Lz1, z[end] - Lz2
    ix1, ix2 = argmin(abs.(x .- x1)), argmin(abs.(x .- x2))
    iy1, iy2 = argmin(abs.(y .- y1)), argmin(abs.(y .- y2))
    iz1, iz2 = argmin(abs.(z .- z1)), argmin(abs.(z .- z2))

    return PML3D(
        ix1, ix2, iy1, iy2, iz1, iz2,
        Kx, Ax, Bx, Ky, Ay, By, Kz, Az, Bz,
        psiExy, psiExz, psiEyx, psiEyz, psiEzx, psiEzy,
        psiHxy, psiHxz, psiHyx, psiHyz, psiHzx, psiHzy,
    )
end


# ******************************************************************************************
# Util
# ******************************************************************************************
function pml_coefficients(x, box, dt; kappa=1, alpha=10e-6, R0=10e-6, m=3)
    Lx1, Lx2 = box
    Nx = length(x)
    xmin, xmax = x[1], x[end]

    eta0 = sqrt(MU0 / EPS0)
    sigma_max1 = -(m + 1) * log(R0) / (2 * eta0 * Lx1)
    sigma_max2 = -(m + 1) * log(R0) / (2 * eta0 * Lx2)

    sigma = zeros(Nx)
    xb1, xb2 = xmin + Lx1, xmax - Lx2
    for ix=1:Nx
        if x[ix] < xb1
            sigma[ix] = sigma_max1 * (abs(x[ix] - xb1) / Lx1)^m
        end
        if x[ix] > xb2
            sigma[ix] = sigma_max2 * (abs(x[ix] - xb2) / Lx2)^m
        end
    end

    K = ones(Nx) * kappa
    B = @. exp(-(sigma / K + alpha) * dt / EPS0)
    A = @. sigma / (sigma * K + alpha * K^2) * (B - 1)

    return K, A, B
end
