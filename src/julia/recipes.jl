abstract type Mult end
struct ByOne <: Mult end
BYONE = ByOne()
struct ByHalf <: Mult end
BYHALF = ByHalf()

(::ByOne)(a,b) = a*b
(::ByHalf)(a,b) = a*b/2

abstract type Perm{a,b,c} end

function is_valid_permutation(a,b,c)
    return (a+b+c==6) & (a == xor(b,c))
end

function permute(triplet::T,perm) where T
    @assert length(triplet) == 3
    return triplet[collect(perm)] :: T
end

Base.collect(axes::A) where {A<:Axes} = [axes.ax1,axes.ax2,axes.ax3]


struct VFrecipe{S<:SCHEME,onleft,T,O,A<:Axes,a,b,c}
    axes::A
    order::O
    q::Vector{T}
    U_at_q::Vector{T}
    checksize::Bool
end

struct DFrecipe{S<:SCHEME,T,O,A<:Axes,AX1}
    axes::A
    order::O
    flux::Vector{T}
    ax1:: AX1
end

struct IPrecipe{S<:SCHEME,T,O,AX<:Axes,AX1,COEF}
    axes::AX
    order::O
    qm::Vector{T}
    ax1:: AX1
end

function VFrecipe(T, axes::A,perm,scheme,maxorder,checksize=false) where {A<:Axes}
    a,b,c = perm
    @assert is_valid_permutation(a,b,c)

    permuted_axes = Axes(permute(collect(axes),perm)...)

    # work pencils to be sent in rightpencil
    n = size(permuted_axes.ax2,EDGES)
    q = zeros(T,n)
    U_at_q = zeros(T,n)

    order = Val(maxorder)
    O = typeof(order)
    newA = typeof(permuted_axes)
    onleft = Val(a<b)

    VFrecipe{scheme,onleft,T,O,newA,a,b,c}(permuted_axes,order,q,U_at_q,checksize)
end

function DFrecipe(T, axes::A,perm,scheme,maxorder) where {A<:Axes}
    a,b,c = perm
    @assert is_valid_permutation(a,b,c)

    permuted_axes = Axes(permute(collect(axes),perm)...)

    n = size(permuted_axes.ax1,EDGES)
    flux = zeros(T,n)

    order = Val(maxorder)
    newA = typeof(permuted_axes)
    O = typeof(order)

    ax1 = Val(perm[1])
    AX1 = typeof(ax1)

    DFrecipe{scheme,T,O,newA,AX1}(permuted_axes,order,flux,ax1)
end


function IPrecipe(T, axes::A,perm,scheme,maxorder,::COEF) where {A<:Axes,COEF<:Mult}
    a,b,c = perm
    @assert is_valid_permutation(a,b,c)

    permuted_axes = Axes(permute(collect(axes),perm)...)

    n = size(permuted_axes.ax1,EDGES)
    qm = zeros(T,n)

    order = Val(maxorder)
    newA = typeof(permuted_axes)
    O = typeof(order)

    ax1 = Val(perm[1])
    AX1 = typeof(ax1)

    IPrecipe{scheme,T,O,newA,AX1,COEF}(permuted_axes,order,qm,ax1)
end

# 2D case
function VFrecipes(u::V,axes::AX,scheme,maxorder) where {T,AX<:Axes,
  A<:Array{T,3},
  V<:Tuple{A,A}}
     return (VFrecipe(T, axes, (1,2,3), scheme, maxorder,true),
             VFrecipe(T, axes, (2,1,3), scheme, maxorder,true))
end

# 3D case
function VFrecipes(u::V,axes::AX,scheme,maxorder) where {T,AX<:Axes,
  A<:Array{T,3},
  V<:Tuple{A,A,A}}
     components = [1,2,3]

     recipes = Dict{Tuple{Integer,Integer}, VFrecipe}()
     for ax1 in components
         for ax2 in setdiff(components, ax1)
             ax3 = setdiff(components, [ax1, ax2])[1]
             #println("$ax1  $ax2  $ax3")
             recipes[(ax1,ax2)] = VFrecipe(T, axes, (ax1,ax2,ax3),
             				   scheme, maxorder,true)

         end
     end
     return recipes
end

function is_valid_size_arrays(du::A,v::A,omega::A,recipe::R)::Bool where {S,onleft,T,O,AX,a,b,c,
                                                                    R<:VFrecipe{S,onleft,T,O,AX,a,b,c},
                                                                    A<:Array{T,3}}
    (;axes) = recipe
    perm = (a,b,c)

    su = size(axes,[EDGES,CENTERS,CENTERS])
    sv = size(axes,[CENTERS,EDGES,CENTERS])
    so = size(axes,[EDGES,EDGES,CENTERS])

    return ( (permute(size(du),perm) == su)
             & (permute(size(v),perm) == sv)
             & (permute(size(omega),perm) == so))
end
