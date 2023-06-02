abstract type ARCH{T} end
struct CPU{T} <: ARCH{T} end
struct GPU{T} <: ARCH{T} end
CPU() = CPU{Float64}()
GPU() = GPU{Float32}()


adapt_storage(::ARCH{T}, x::AbstractFloat) where T = T(x)
adapt_storage(::ARCH{T}, x::Complex) where T = Complex{T}(x)
adapt_storage(::ARCH{T}, x::StepRangeLen) where T = range(T(first(x)), T(last(x)), x.len)
adapt_storage(::CPU{T}, x::Array) where T = Array{T}(x)
adapt_storage(::GPU{T}, x::Array) where T = CuArray{T}(x)
adapt_storage(::CPU{T}, x::Array{TA}) where {T, TA<:Complex} = Array{Complex{T}}(x)
adapt_storage(::GPU{T}, x::Array{TA}) where {T, TA<:Complex} = CuArray{Complex{T}}(x)


# Arrays of indices:
adapt_storage(::CPU{T}, x::Array{TA}) where {T, TA<:Int} = Array{Int64}(x)
adapt_storage(::GPU{T}, x::Array{TA}) where {T, TA<:Int} = CuArray{Int32}(x)
adapt_storage(::CPU, x::Vector{<:CartesianIndex}) = x
adapt_storage(::GPU, x::Vector{<:CartesianIndex}) = CuArray(x)


# function adapt_storage(::CPU{T}, p::cFFTWPlan) where T
#     tmp = zeros(Complex{T}, p.sz)
#     return plan_fft!(tmp, p.region)
# end

# function adapt_storage(::GPU{T}, p::cFFTWPlan) where T
#     tmp = CUDA.zeros(Complex{T}, p.sz)
#     return plan_fft!(tmp, p.region)
# end


macro krun(ex...)
    N = ex[1]
    call = ex[2]
    args = call.args[2:end]
    @gensym kernel config threads blocks
    code = quote
        local $kernel = @cuda launch=false $call
        local $config = launch_configuration($kernel.fun)
        local $threads = min($config.threads, $N)
        local $blocks = min($config.blocks, cld($N, $threads))
        $kernel($(args...); threads=$threads, blocks=$blocks)
    end
    return esc(code)
end
