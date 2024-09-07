module CFTransport

using MutatingOrNot: Void
using ManagedLoops: @loops, @vec
using CFDomains: HVLayout, VHLayout, PressureCoordinate, nlayer, mass_level

export GodunovScheme, VanLeerScheme
export concentrations!, slopes!, fluxes!, FV_update!

macro fast(code)
    debug = haskey(ENV, "GF_DEBUG") && (ENV["GF_DEBUG"]!="")
    return debug ? esc(code) : esc(quote @inbounds $code end)
end

"""
    abstract type OneDimOp{dim,rank} end

One-dimensional operator acting on dimension `dim` of arrays of rank `rank`.
"""
abstract type OneDimOp{Dim,Rank} end

include("julia/stencil.jl")

@inline half(x) = @fastmath x/2

include("julia/finite_volume.jl")
include("julia/godunov.jl")
include("julia/vanleer.jl")
include("julia/limiters.jl")
include("julia/remap_fluxes.jl")

end # module
