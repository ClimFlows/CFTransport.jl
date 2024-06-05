"""
    abstract type Stencil <: Function end

A one-dimensional stencil is a function/closure/callable of the form

    function stencil((i,j), params, arrays)
        a, b, ... = arrays
        a[i,j] = expression(params, b[i,j], b[i+1,j], b[i-1,j], ...)
    end
"""
abstract type Stencil <: Function end

"""
    stencil = expand_stencil(dim, rank, stencil1)
Return a `stencil` of rank `rank` that applies the one-dimensional
stencil `stencil1` to dimension `dim`. For example:

    function stencil((i,j), params, arrays)
        a, b = arrays
        a[i,j] = expression(params, b[i,j], b[i+1,j], b[i-1,j])
    end

    stencil2 = expand_stencil(1, 2, stencil1)
    for i in axes(a,1), j in axes(a,2)
        stencil2((i,j), params, (a,b))
    end

is equivalent to:

    for i in axes(a,1), j in axes(a,2)
        a[i,j] = expression(params, b[i,j], b[i,j+1], b[i,j-1])
    end

See also [`Stencil`](@ref)
"""
struct expand_stencil{Index, Indices, Fun} <: Function
    call::Fun
    @inline expand_stencil{index,rank}(call::Fun) where {index, rank, Fun<:Function} = new{index, rank, Fun}(call)
end

@inline function (stencil::expand_stencil)(ijk, arg, arrays)
    ij, arrays = slices(stencil, ijk, arrays)
    @inline stencil.call(ij, arg, arrays)
end

const xps=expand_stencil

struct Permute{dim,rank,A}
    array::A
    Permute{dim, rank}(array::A) where {dim, rank, A} = new{dim, rank, A}(array)
end

slices(::xps{1,1}, i, arrays)     = (i,nothing), map(Permute{1,1}, arrays)
@Base.propagate_inbounds Base.getindex(p::Permute{1,1}, i, ::Nothing) = p.array[i]
@Base.propagate_inbounds Base.setindex!(p::Permute{1,1}, val, i, ::Nothing) = (p.array[i]=val)

slices(::xps{1,2}, (i,j), arrays) = (i,j), map(Permute{1,2}, arrays)
@Base.propagate_inbounds Base.getindex(p::Permute{1,2}, i, j) = p.array[i,j]
@Base.propagate_inbounds Base.setindex!(p::Permute{1,2}, val, i, j) = (p.array[i,j]=val)

slices(::xps{2,2}, (i,j), arrays) = (j,i), map(Permute{2,2}, arrays)
@Base.propagate_inbounds Base.getindex(p::Permute{2,2}, i, j) = p.array[j,i]
@Base.propagate_inbounds Base.setindex!(p::Permute{2,2}, val, i, j) = (p.array[j,i]=val)

#=

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

=#
