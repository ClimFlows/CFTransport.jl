"""
A one-dimensional stencil is a function/closure/callable of the form

    function stencil(i, params, arrays)
        a, b, ... = arrays
        a[i] = expression(params, b[i], b[i+1], b[i-1], ...)
    end

A two-dimensional stencil is a function/closure/callable of the form

    function stencil((i,j), params, arrays)
        a, b, ... = arrays
        a[i,j] = expression(params, b[i,j], b[i+1,j], b[i,j+1], ...)
    end

A stencil acts at a single index. For performance, stencils should be @inline and use @inbounds.
"""
abstract type Stencil <: Function end

"""
    stencil = expand_stencil(dim, rank, stencil1)

Returns a `stencil` of rank `rank` that applies the one-dimensional
stencil `stencil1` to dimension `dim`.

    stencil = apply_stencil(dims, rank, stencilN)

Returns a `stencil` of rank `rank` that applies the `N`-dimensional
stencil `stencilN` to the `N`-tuple of dimensions `dims`.

`stencil` can be passed to `forall` :

    forall(stencil, backend, ranges, arrays)

where `arrays` is the tuple of arrays on which the stencil acts
and `ranges` is more or less `axes(a)` with `a` one of the arrays,
except that the range over the stencil index shoud be reduced
according to the stencil width to avoid out-of-bounds accesses.

See also [`Stencil`](@ref)
"""
struct expand_stencil{Index, Indices, Fun} <: Function
    call::Fun
    @inline expand_stencil{index,rank}(call::Fun) where {index, rank, Fun<:Function} = new{index, rank, Fun}(call)
end

@inline function (stencil::expand_stencil)(i, arg, arrays)
    i, arrays = @inline slices(stencil, i, arrays)
    @inline stencil.call(i, arg, arrays)
end

const xps=expand_stencil

slices(::xps{1,1}, i, arrays)       = i, arrays

slices(::xps{1,2}, (i,j), arrays)   = i, map( x->(@inbounds @view x[:,j]),   arrays)
slices(::xps{2,2}, (i,j), arrays)   = j, map( x->(@inbounds @view x[i,:]),   arrays)

slices(::xps{1,3}, (i,j,k), arrays) = i, map( x->(@inbounds @view x[:,j,k]), arrays)
slices(::xps{2,3}, (i,j,k), arrays) = j, map( x->(@inbounds @view x[i,:,k]), arrays)
slices(::xps{3,3}, (i,j,k), arrays) = k, map( x->(@inbounds @view x[i,j,:]), arrays)

struct ApplyStencil{Index, Indices, S} <: Function
    stencil::S
end

"""
    kernel = apply_stencil(dim, rank, stencil)

Returns a `kernel` that applies a one-dimensional stencil
to dimension `dim` of input/output arrays of rank `rank`

    kernel = apply_stencil(dims, rank, stencil)

Returns a `kernel` that applies a multi-dimensional stencil
to dimensions 'dims' of input/output arrays of rank `rank`.

`kernel` can be called directly or offloaded to a backend :

    kernel(ranges, arrays)
    (kernel=>backend)(ranges, arrays)

where `ranges` is more or less `axes(a)` with `a` one of the arrays,
except that the range over the stencil index shoud be reduced
according to the stencil width to avoid out-of-bounds accesses.

See also [`Stencil`](@ref)
"""
apply_stencil(index, indices, stencil::S) where S = ApplyStencil{index, indices, S}(stencil)

# Apply 1D stencil to first dimension of 2-dimensional arrays
@inline function (kernel::ApplyStencil{1, 2})((irange, jrange), arrays)
    for j in jrange
        @simd for i in irange
            slice(x) = @inbounds @view x[:,j]
            kernel.stencil(i, map(slice, arrays))
        end
    end
    nothing
end

# Apply 1D stencil to second dimension of 2-dimensional arrays
@inline function (kernel::ApplyStencil{2, 2})((irange, jrange), arrays)
    for j in jrange
        @simd for i in irange
            slice(x) = @inbounds @view x[i,:]
            kernel.stencil(j, map(slice, arrays))
        end
    end
    nothing
end
