using CFTransport: UP,CE,WZ,WJS
using CFTransport: AXES,Axes,Axis,HALOED,CLOSED,EMPTY,CENTERS,EDGES
using CFTransport: leftpencil,rightpencil
using CFTransport: VFrecipe,DFrecipe,IPrecipe,BYONE,BYHALF

getq() = [1.0, 12.0, 5.0, 3.0, 15.0, 7.0]

function test_scaling_invariance(scheme)
    coef = 1e-30
    U = 1.0
    q = getq()
    for n in [2,4,6]
        res1 = scheme(U, coef*q[1:n]...)/coef
        res2 = scheme(U, q[1:n]...)
        @test isapprox(res1, res2)
    end
end

function test_upwinding(scheme)
    q = getq()
    for n in [2,4,6]
        res1 = scheme(1.0, q[1:n]...)
        res2 = scheme(-1.0, q[1:n]...)
        @test res1 isa eltype(q)
        @test res2 isa eltype(q)
    end
end

function test_type(scheme)
    for T in [Float16, Float32, Float64]
        q = convert.(T,getq())
        for n in [2,4,6]
            out = scheme(T(1.0), q[1:n]...)
            @test out isa T
        end
    end
end

function test_values(scheme)
    q = [1.0, 12.0, 5.0, 3.0, 15.0, 7.0]
    values = Dict(UP=>(1.0,11.5,1.95),
                  CE=>(6.5,9.583333333333332,1.4666666666666666),
                  WJS=>(1.0,9.767984978288933,3.1397459255338642),
                  WZ=>(1.0,10.697088205684194,2.7751506042917278)
                  )

    for n in [2,4,6]
        res = scheme(1.0,q[1:n]...)
        @test isapprox(res, values[scheme][div(n,2)])
    end
end

function test_schemes()
    for scheme in [UP,CE,WJS,WZ]
        test_type(scheme)
        test_upwinding(scheme)
        test_values(scheme)
        test_scaling_invariance(scheme)
    end
end


function test_axes()
    nx,ny,nz,nhalo = 4, 5, 3, 2
    axi = Axis(HALOED, nx, nhalo)
    axj = Axis(CLOSED, ny, nhalo)
    axk = Axis(EMPTY, nz, nhalo)
    @test size(axi,CENTERS) == nx+2nhalo
    @test size(axi,EDGES) == nx+2nhalo
    @test size(axj,CENTERS) == ny
    @test size(axj,EDGES) == ny+1
    @test size(axk,CENTERS) == 1
    @test size(axk,EDGES) == 1

    @test length(CENTERS(axi)*CENTERS(axj))==nx*ny
    @test length(EDGES(axj)*CENTERS(axk))==(ny+1)

    axes = AXES(axi,axj,axk)
    @test axes == Axes(nx,ny,nz,nhalo, HALOED, CLOSED,EMPTY)
end

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
                @test all(flx[1:nhalo] .== 0)
                @test all(flx[end-nhalo+2:end] .== 0)
                @test all(flx[nhalo+1:end-nhalo+1] .> 0)

            elseif T <: CLOSED
                @test (flx[1] ==0) & (flx[end] ==0)
                @test all(flx[2:end-1] .> 0)
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
        @test all(@. flx0 == flx*coef)
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
                @test all(@. dq[nhalo+1:nhalo+n] > 0)
                @test all(@. dq[1:nhalo] == 0)
                @test all(@. dq[nhalo+n+1:end] == 0)

            elseif T <: CLOSED
                @test all(@. dq > 0)
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
        @test all(@. dq0 == dq)
    end

    # test that the term is added to dq
    order = 6
    dq*=0
    dq0 = dq*0
    rightpencil(dq,q,U,Val(order),ax,scheme)
    rightpencil(dq,q,U,Val(order),ax,scheme)
    rightpencil(dq0,q,U,Val(order),ax,scheme)
    @test all(@. 2*dq0 == dq)

end

function test_pencils()
    for T in [HALOED,CLOSED]
        test_leftpencils(T)
        test_rightpencils(T)
    end
end

function test_recipes()
    nx,ny,nz,nhalo = 2, 8, 4, 3
    axi = Axis(CLOSED, nx, nhalo)
    axj = Axis(HALOED, ny, nhalo)
    axk = Axis(CLOSED, nz, nhalo)
    axes = AXES(axi,axj,axk)

    scheme = UP
    maxorder = 6
    #uvrecipe = VFrecipe(axes,permutation(1,2,3),scheme,maxorder)
    #uwrecipe = VFrecipe(axes,permutation(1,3,2),scheme,maxorder)

    vfrecipe = VFrecipe(Float64, axes,(1,3,2), scheme, maxorder)
    @test typeof(vfrecipe)<:VFrecipe

    dfrecipe = DFrecipe(Float64, axes,(1,3,2), scheme, maxorder)
    @test typeof(dfrecipe)<:DFrecipe

    iprecipe = IPrecipe(Float64, axes,(1,3,2), scheme, maxorder, BYONE)
    @test typeof(iprecipe)<:IPrecipe

end


@testset "WENO" begin
    test_schemes()
    test_axes()
    test_pencils()
    test_recipes()
end
