import Adapt: @adapt_structure
import DelimitedFiles: readdlm


struct IonizationRate{T, A<:AbstractVector{T}} <: Function
    x :: A
    y :: A
end

@adapt_structure IonizationRate


function IonizationRate(fname::String)
    data = readdlm(fname)
    x = data[:,1]
    y = data[:,2]

    @. x = log10(x)
    @. y = log10(y)

    # Check for sorted and evenly spaced x values:
    dx = diff(x)
    @assert issorted(x)
    @assert all(x -> isapprox(x, dx[1]), dx)

    return IonizationRate(x, y)
end


function (tf::IonizationRate{T})(x::T) where T
    if x <= 0
        y = zero(T)   # in order to avoid -Inf in log10(0)
    else
        xlog10 = log10(x)
        ylog10 = linterp(xlog10, tf.x, tf.y)
        y = 10^ylog10
    end
    return y
end


function linterp(xi, x, y)
    if xi <= x[1]
        i = 1
    elseif xi >= x[end]
        i = length(x) - 1
    else
        i = searchsortedfirst(x, xi) - 1
    end
    dydx = (y[i+1] - y[i]) / (x[i+1] - x[i])
    return y[i] + dydx * (xi - x[i])
end
