module CFTransport

using MutatingOrNot: Void
using ManagedLoops: @loops, @vec
using CFDomains: HVLayout, VHLayout, PressureCoordinate, nlayer, mass_level

export GodunovScheme, VanLeerScheme
export concentrations!, slopes!, fluxes!, FV_update!

macro fast(code)
#    debug = haskey(ENV, "GF_DEBUG") && (ENV["GF_DEBUG"]!="")
    debug = true
    return debug ? esc(code) : esc(quote @inbounds $code end)
end

include("julia/stencil.jl")

"""
    abstract type OneDimOp{dim,rank} end

One-dimensional operator acting on dimension `dim` of arrays of rank `rank`.
"""
abstract type OneDimOp{Dim,Rank} end

# excludes from range `1:N` `a` items at the start, and `b` items at the end,
struct Crop{a,b} end
crop(a,b)=Crop{a,b}()
(::Crop{a,b})(N::Int) where {a,b} = (1+a):(N-b)

# enforces restriction `r` on dimension `k` :
# replaces dimension `dim_k` by the result of r(size(dim_k)),
# e.g. 1:N => 2:N-1
@inline restrict(::OneDimOp{1,1}, r::Fun, (dim,) )      where Fun =  r(length(dim))
@inline restrict(::OneDimOp{1,2}, r::Fun, (dim1,dim2) ) where Fun = (r(length(dim1)), dim2)
@inline restrict(::OneDimOp{2,2}, r::Fun, (dim1,dim2) ) where Fun = (dim1, r(length(dim2)))

# applies function `step` related to operator `op` to `arrays` on `backend`,
# after restricting dimension `dim` to only(size(dim))
@inline function invoke(step::Fun, op::OneDimOp{dim,2}, backend, ranges, only::Only, arrays) where {dim,Fun,Only}
    step = expand_stencil{dim, 2}(step)
    ranges = restrict(op, only, ranges)
    invoke_step(backend, ranges, step, op, arrays)
end

@loops function invoke_step(_, ranges, step, op, arrays)
    let (ri,rj) = ranges
        for j in rj
            @vec for i in ri
                @inline step((i,j), op, arrays)
            end
        end
    end
end

@inline half(x) = @fastmath x/2

include("julia/finite_volume.jl")
include("julia/godunov.jl")
include("julia/vanleer.jl")
include("julia/limiters.jl")
include("julia/remap_fluxes.jl")

end # module
