include("axes.jl")
include("schemes.jl")

include("leftpencils.jl")
include("rightpencils.jl")

Base.view(::Val{A},ind...) where A = Val(A)


function is_valid_rightpencil_arrays(du, q, Um, order, ax)
    println( (length(du), size(ax,CENTERS)),
        (length(q), size(ax,EDGES)),
             (length(Um), size(ax,EDGES)))

    return ( (length(du)==size(ax,CENTERS)) &
        (length(q)==size(ax,EDGES)) &
        (length(Um)==size(ax,EDGES)) )
end

# ------------------------- end of the module ------------------------------------
# TESTS

# TODO : test local orders (orders are pencils, not Val(order))

function test_leftpencils(T)
    n,nhalo = 8,3
    orders = 2:2:6
    ax = Axis(T,n,nhalo)

    q = zeros(size(ax,CENTERS))
    U = zeros(size(ax,EDGES))
    flx = zeros(size(ax,EDGES))

    q .= (1:length(q)).^2
    @. U = 1
    for scheme in [UP,CE,WZ,WJS]
        for order in orders
            flx .= 0
            leftpencil(flx,q,U,Val(order),ax,scheme)
            if T <: HALOED
                @assert all(flx[1:nhalo] .== 0)
                @assert all(flx[end-nhalo+2:end] .== 0)
                @assert all(flx[nhalo+1:end-nhalo+1] .> 0)

            elseif T <: CLOSED
                @assert (flx[1] ==0) & (flx[end] ==0)
                @assert all(flx[2:end-1] .> 0)
            end
        end
    end

    # test that flx is linear in U
    scheme = UP
    flx0 = flx*0
    coef=10.0
    for order in orders
        leftpencil(flx,q,U,Val(order),ax,scheme)
        leftpencil(flx0,q,U*coef,Val(order),ax,scheme)
        @assert all(@. flx0 == flx*coef)
    end
end

function test_rightpencils(T)
    n,nhalo = 8,3
    orders = 2:2:6
    ax = Axis(T,n,nhalo)

    q = zeros(size(ax,EDGES))
    U = zeros(size(ax,EDGES))
    dq = zeros(size(ax,CENTERS))

    q .= (1:length(q)).^2
    @. U = 1

    for scheme in [UP,CE,WZ,WJS]
        for order in orders
            @. dq = 0
            rightpencil(dq,q,U,Val(order),ax,scheme)
            if T <: HALOED
                @assert all(@. dq[nhalo+1:nhalo+n] > 0)
                @assert all(@. dq[1:nhalo] == 0)
                @assert all(@. dq[nhalo+n+1:end] == 0)

            elseif T <: CLOSED
                @assert all(@. dq > 0)
            end
        end
    end

    # test that dq does not depend on U (apart from the sign for upwinded schemes)
    scheme = UP
    dq*=0
    dq0 = dq*0
    coef=10.0
    for order in orders
        rightpencil(dq,q,U,Val(order),ax,scheme)
        rightpencil(dq0,q,U*10,Val(order),ax,scheme)
        @assert all(@. dq0 == dq)
    end

    # test that the term is added to dq
    order = 6
    dq*=0
    dq0 = dq*0
    rightpencil(dq,q,U,Val(order),ax,scheme)
    rightpencil(dq,q,U,Val(order),ax,scheme)
    rightpencil(dq0,q,U,Val(order),ax,scheme)
    @assert all(@. 2*dq0 == dq)

end

function test_pencils()
    for T in [HALOED,CLOSED]
        test_leftpencils(T)
        test_rightpencils(T)
    end
end

# nx,ny,nz = shape = (100,100,100)
# nhalo = 2

# q = zeros(nx,ny,nz)
# U = zeros(nx+1,ny,nz)
# dU = zeros(nx+1,ny,nz)
# V = zeros(nx,ny+1,nz)
# dV = zeros(nx,ny+1,nz)
# W = zeros(nx,ny,nz+1)
# dW = zeros(nx,ny,nz+1)

# U.=randn(size(U))
# V.=randn(size(V))
# W.=randn(size(W))
# q.=randn(size(q))

# axi = Axis(CLOSED,nx,nhalo)
# axj = Axis(CLOSED,ny,nhalo)
# axk = Axis(CLOSED,nz,nhalo)

# order = Val(6)

# i,j,k = 34, 13, 50

# leftpencil(view(dU,:,j,k),view(q,:,j,k),view(U,:,j,k),Val(6),axi,UP)

# @btime leftpencil(view($dU,:,$j,$k),view($q,:,$j,$k),view($U,:,$j,$k),view($order,:,$j,$k),$axi,WZ)
# @btime leftpencil(view($dV,$i,:,$k),view($q,$i,:,$k),view($V,$i,:,$k),Val(6),$axj,WZ)
# @btime leftpencil(view($dW,$i,$j,:),view($q,$i,$j,:),view($W,$i,$j,:),Val(6),$axk,WZ)
