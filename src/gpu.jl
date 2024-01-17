abstract type Backend{T} end


"""
    CPU{T}()

CPU backend. CPU{Float32}() for single precision and CPU{Float64}() for double precision
floats. CPU(), without the T parameter, is equivalent to CPU{Float64}().
"""
struct CPU{T} <: Backend{T} end


"""
    GPU{T}()

CUDA backend. GPU{Float32}() for single precision and GPU{Float64}() for double precision
floats. GPU(), without the T parameter, is equivalent to GPU{Float32}().
"""
struct GPU{T} <: Backend{T} end


CPU() = CPU{Float64}()
GPU() = GPU{Float32}()


# Numbers:
adapt_storage(::Backend{T}, x::AbstractFloat) where T = T(x)
adapt_storage(::Backend{T}, x::Complex) where T = Complex{T}(x)

# Range:
adapt_storage(::Backend{T}, x::StepRangeLen) where T = range(T(first(x)), T(last(x)), x.len)

# Arrays:
adapt_storage(::CPU{T}, x::Array) where T = Array{T}(x)
adapt_storage(::GPU{T}, x::Array) where T = CuArray{T}(x)
adapt_storage(::CPU, x::Array{<:Bool}) = Array{Bool}(x)
adapt_storage(::GPU, x::Array{<:Bool}) = CuArray{Bool}(x)
adapt_storage(::CPU, x::Array{<:Integer}) = Array{Int64}(x)
adapt_storage(::GPU, x::Array{<:Integer}) = CuArray{Int32}(x)
adapt_storage(::CPU, x::Vector{<:CartesianIndex}) = x
adapt_storage(::GPU, x::Vector{<:CartesianIndex}) = CuArray(x)
