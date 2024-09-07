# Machinery to apply a 1D stencil operator to dimension `dim` of arrays of rank `rank`.
# To handle the different combinations of {dim, rank}, the following lower-level features are combined:
#    * invoke(...): loop nest over `rank` indices
#    * stencil = expand_stencil(dim rank, stancil) : callable object wich applies the 1D stencil at a single index `ijk` of higher-rank arrays.
#    * Perm{dim, rank, A<:Array} : wraps an array to change the order of indices. 1D stencil operates on such wrapped arrays.
#    * permute(...) : applies the appropriate permutation to an index (i,j,...) and to arrays
# invoke(...) -> (stencil)(...) -> permute
#                               -> Perm
#                               -> 1D operator -> getindex/setindex!

# Examples:

# `invoke` passes `xp = xps{2,2}(fun)` to `invoke_step2`
# `invoke_step2` loops on `i,j`, passes (i,j) to xp
# xp :
#  * passes (i,j) to permute, receives (j,i)
#  * calls fun at index (j,i) on map(p, arrays) where p = Perm{2,2}
# fun :
#  * gets (j,i) and map(p,arrays)
#  * gets/sets array[j,i]
# p(array)[j,i] = array[i,j]

# invoke passes xp = xps{3,3}(fun) to invoke_step3
# invoke_step3 loops on (i,j,k), passes (i,j,k) to xp
# xp :
#  * passes (i,j,k) to permute, receives (k,(i,j))
#  * calls fun at index (k,(i,j)) on map(p, arrays) where p = Perm{3,3}
# fun :
#  * gets k, (i,j) and map(p,arrays)
#  * gets/sets array[k±1, (i,j)]
# p(array)[k±1,(i,j)] = array[i,j,k±1]


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
struct expand_stencil{Index,Indices,Fun} <: Function
    call::Fun
    @inline expand_stencil{index,rank}(call::Fun) where {index,rank,Fun<:Function} =
        new{index,rank,Fun}(call)
end
const xps = expand_stencil

struct Perm{dim,rank,A}
    array::A
    Perm{dim,rank}(array::A) where {dim,rank,A} = new{dim,rank,A}(array)
end

#============================= invoke ============================#

# applies function `step` related to operator `op` to `arrays` on `backend`,
# after restricting dimension `dim` to only(size(dim))

@inline function invoke(
    step::Fun,
    op::OneDimOp{dim,rank},
    backend,
    ranges,
    only::Only,
    arrays,
) where {dim,rank,Fun,Only}
    step = expand_stencil{dim,rank}(step)
    ranges = restrict(op, only, ranges)
    invoke_step(backend, ranges, step, op, arrays)
end

invoke_step(backend, ranges, step, op::OneDimOp{dim,2}, arrays) where {dim} =
    invoke_step2(backend, ranges, step, op, arrays)

invoke_step(backend, ranges, step, op::OneDimOp{dim,3}, arrays) where {dim} =
    invoke_step3(backend, ranges, step, op, arrays)

@loops function invoke_step2(_, ranges, step, op, arrays)
    let (ri, rj) = ranges
        for j in rj
            @vec for i in ri
                @inline step((i, j), op, arrays)
            end
        end
    end
end

@loops function invoke_step3(_, ranges, step, op, arrays)
    let (ri, rj, rk) = ranges
        for j in rj, k in rk
            @vec for i in ri
                @inline step((i, j, k), op, arrays)
            end
        end
    end
end

# excludes from range `1:N` `a` items at the start, and `b` items at the end,
struct Crop{a,b} end
crop(a, b) = Crop{a,b}()
(::Crop{a,b})(N::Int) where {a,b} = (1+a):(N-b)

# enforces restriction `r` on dimension `k` :
# replaces dimension `dim_k` by the result of r(size(dim_k)),
# e.g. 1:N => 2:N-1
restrict(::OneDimOp{1,1}, r::Fun, (dim,)) where {Fun} = r(length(dim))
restrict(::OneDimOp{1,2}, r::Fun, (dim1, dim2)) where {Fun} = (r(length(dim1)), dim2)
restrict(::OneDimOp{2,2}, r::Fun, (dim1, dim2)) where {Fun} = (dim1, r(length(dim2)))
restrict(::OneDimOp{3,3}, r::Fun, (dim1, dim2, dim3)) where {Fun} =
    (dim1, dim2, r(length(dim3)))

#======================================== permute =====================================#

# `permute`` turns index ijk=(i, ...) into a pair (a,b)
#     where a is the index in dimension `dim` while `b` packs the other indices,
#     and applies Perm{dim,rank} to input/output arrays.
# The 1D stencil receives these (permuted) arrays and gets/sets array[a \pm offset, b]
# getindex/setindex! undoes the transformation ijk -> (a,b)

using Base: @propagate_inbounds as @prop
import Base: getindex, setindex!

@inline function (stencil::expand_stencil{dim, rank})(ijk, arg, arrays) where {dim, rank}
    ij, P = permute(stencil, ijk)
    arrays = map(Perm{dim,rank}, arrays)
    @inline stencil.call(ij, arg, arrays)
end

permute(::xps{1,1}, i) = (i, nothing), Perm{1,1}
@prop getindex(p::Perm{1,1}, i, ::Nothing) = p.array[i]
@prop setindex!(p::Perm{1,1}, val, i, ::Nothing) = (p.array[i] = val)

permute(::xps{1,2}, (i, j)) = (i, j), Perm{1,2}
@prop getindex(p::Perm{1,2}, i, j) = p.array[i, j]
@prop setindex!(p::Perm{1,2}, val, i, j) = (p.array[i, j] = val)

permute(::xps{2,2}, (i, j)) = (j, i), Perm{2,2}
@prop getindex(p::Perm{2,2}, j, i) = p.array[i, j]
@prop setindex!(p::Perm{2,2}, val, j, i) = (p.array[i, j] = val)

permute(::xps{1,3}, (i, j, k)) = (i, (j, k)), Perm{1,3}
@prop getindex(p::Perm{1,3}, i, (j, k)) = p.array[i, j, k]
@prop setindex!(p::Perm{1,3}, val, i, (j, k)) = (p.array[i, j, k] = val)

permute(::xps{2,3}, (i, j, k)) = (j, (i, k)), Perm{2,3}
@prop getindex(p::Perm{2,3}, j, (i, k)) = p.array[i, j, k]
@prop setindex!(p::Perm{2,3}, val, j, (i, k)) = (p.array[i, j, k] = val)

permute(::xps{3,3}, (i, j, k)) = (k, (i, j)), Perm{3,3}
@prop getindex(p::Perm{3,3}, k, (i, j)) = p.array[i, j, k]
@prop setindex!(p::Perm{3,3}, val, k, (i, j)) = (p.array[i, j, k] = val)
