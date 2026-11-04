abstract type AXIS end
abstract type HALOED<:AXIS end
abstract type CLOSED<:AXIS end
abstract type NONE<:AXIS end

abstract type LOCATION end
abstract type CENTERS<:LOCATION end
abstract type EDGES<:LOCATION end
abstract type INTERIOREDGES<:LOCATION end
abstract type EMPTY<:LOCATION end

struct Axis{AXIS,T}
    n:: T
    nhalo:: T
end

# constructors
Axis(::Type{HALOED}, n, nhalo) = Axis{HALOED, Int64}(n, nhalo)
Axis(::Type{CLOSED}, n, nhalo) = Axis{CLOSED, Int64}(n, 0)
Axis(::Type{EMPTY}, n, nhalo) = Axis{EMPTY, Int64}(0, 0)

# this type is extensively used in API
struct AXES{A,B,C<:Axis}
    ax1::A
    ax2::B
    ax3::C
end


Axes(nx,ny,nz,nhalo,A,B,C) = AXES(Axis(A,nx,nhalo), Axis(B,ny,nhalo), Axis(C,nz,nhalo))

# size of an array along this axis, the size depends on the location for CLOSED axis
Base.size(a::Axis{HALOED}, ::Type{L}) where {L<:LOCATION} = a.n+2a.nhalo
Base.size(a::Axis{CLOSED}, ::Type{CENTERS}) = a.n
Base.size(a::Axis{CLOSED}, ::Type{EDGES}) = a.n+1
Base.size(a::Axis{EMPTY}, ::Type{L}) where {L<:LOCATION} = 1
Base.size(axes::AXES, location)::Tuple{Int64,Int64,Int64} = Tuple(size(d,l) for (d,l) in zip([axes.ax1,axes.ax2,axes.ax3],location))

# iterators, for the expressiveness of loops and also to avoid computations in the halo
@inline (::Type{CENTERS})(a::Axis{HALOED,T} where {T})=a.nhalo+1:a.nhalo+a.n
@inline (::Type{EDGES})(a::Axis{HALOED,T} where {T})=a.nhalo+1:a.nhalo+a.n+1
@inline (::Type{INTERIOREDGES})(a::Axis{HALOED,T} where {T})=a.nhalo+1:a.nhalo+a.n+1
@inline (::Type{CENTERS})(a::Axis{CLOSED,T} where {T})=1:a.n
@inline (::Type{EDGES})(a::Axis{CLOSED,T} where {T})=1:a.n+1
@inline (::Type{INTERIOREDGES})(a::Axis{CLOSED,T} where {T})=2:a.n
@inline (::Type{L} where {L<:LOCATION})(a::Axis{EMPTY,T} where {T})=1

# sugar coating for double loops
Base.:*(a::UnitRange,b::UnitRange) = Iterators.product(a,b)


