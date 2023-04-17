function pml(x, box, dt; kappa=1, alpha=10e-6, R0=10e-6, m=3)
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
