include("pencils.jl")
include("recipes.jl")


signature(a,b,c) = xor(iseven(a),(b<c))


"""
  `vortex_force_oneterm(du, v, omega, recipe)`

  compute the term (aka vortex force, or interior product)

  du(ax1) += signature(ax1,ax2,ax3) * v(ax2) * omega(ax3)

  where (ax1, ax2, ax3) is a permutation of (1,2,3) and indicates the
  axes of each component.

  The method uses the `scheme<:SCHEME` to reconstruct v*omega. The
  permutation and the `scheme` are stored in `recipe<:VFrecipe`.

"""
function vortex_force_oneterm(du::A,
                              v::A,
                              omega::A,
                              recipe::R) where {S,onleft,T,O,AX,a,b,c,
                                                R<:VFrecipe{S,onleft,T,O,AX,a,b,c},
                                                A<:Array{T,3}}

    (;q,U_at_q,order,axes,checksize) = recipe


    if checksize
        # TODO: fix the allocations arising this call
        value = is_valid_size_arrays(du,v,omega,recipe)
        @assert value
    end

    applysign = signature(a,b,c) ? x -> x : x -> -x

    for i in INTERIOREDGES(axes.ax1), k in CENTERS(axes.ax3)

        average(U_at_q, view2d(v,k,Val(c)), axes.ax2, i, onleft)
        om = view1d(omega,i,k,Val(b))
        @. q = applysign(U_at_q*om)

        rightpencil(view1d(du,i,k,Val(b)),
                    q, U_at_q,
                    view1d(order,i,k,Val(b)),
                    axes.ax2, S)

    end
    return
end

"""
  `divflux_oneterm(dq, U, q, recipe)`

  compute the divergence of the tracer flux

  dq -= d(U(ax1)*q)/dax1

  where `ax1` is the axis on which the flux is computed

  The method uses the `scheme<:SCHEME` to reconstruct q. `ax1` and
  `scheme` are stored in `recipe<:DFrecipe`.

"""
function divflux_oneterm(dq::A,
                         U::A,
                         q::A,
                         recipe::R) where {S,T,O,AX,AX1,
                                           R<:DFrecipe{S,T,O,AX,AX1},
                                           A<:Array{T,3}}

    (;flux,order,axes,ax1) = recipe

    for j in CENTERS(axes.ax2), k in CENTERS(axes.ax3)

        leftpencil(flux,
                   view1d(q,j,k,ax1),
                   view1d(U,j,k,ax1),
                   view1d(order,j,k,ax1),
                   axes.ax1, S)

        dq1d = view1d(dq,j,k,ax1)

        for i in CENTERS(axes.ax1)
            dq1d[i] += flux[i]-flux[i+1]
        end

    end
    nothing
end

"""
  `innerproduct_oneterm(q, u, U, recipe)`

  compute the innerproduct

  q += u(ax1)*U(ax1)*coef

  where `ax1` is the axis on which the product is done, and `coef` is
  a predefined coefficient, basically 1 or 0.5. The 0.5 coefficient is
  used to compute the kinetic energy.

  The method uses the `scheme<:SCHEME` to reconstruct q. `ax1`, `coef` and
  `scheme` are stored in `recipe<:IPrecipe`.

"""
function innerproduct_oneterm(q::A,
                              u::A,
                              U::A,
                              recipe::R) where {S,T,O,AX,AX1,COEF,
                                                R<:IPrecipe{S,T,O,AX,AX1,COEF},
                                                A<:Array{T,3}}

    (;qm,order,axes,ax1) = recipe

    for j in CENTERS(axes.ax2), k in CENTERS(axes.ax3)

        q1d = view1d(q,j,k,ax1)
        u1d = view1d(u,j,k,ax1)
        U1d = view1d(U,j,k,ax1)

        #println("$(size(u1d))  $(size(U1d))   $(size(qm))")

        @. qm = u1d*U1d

        rightpencil(q1d,
                    qm,
                    U1d,
                    view1d(order,j,k,ax1),
                    axes.ax1,
                    S)

    end
    return
end

(::BYONE)(a,b) = a*b
(::BYHALF)(a,b) = a*b/2


# (::Val{true})(x) = x
# (::Val{false})(x) = -x

view2d(q,k,::Val{3}) = view(q,:,:,k)
view2d(q,k,::Val{2}) = view(q,:,k,:)
view2d(q,k,::Val{1}) = view(q,k,:,:)

view1d(q,i,j,::Val{3}) = view(q,i,j,:)
view1d(q,i,j,::Val{2}) = view(q,i,:,j)
view1d(q,i,j,::Val{1}) = view(q,:,i,j)

Base.view(::Val{N},ind...) where N = Val(N)



function average(Um, v, axj::Axis, i, ::Val{true})
    for j in EDGES(axj)
        Um[j] = (v[i,j]+v[i-1,j])*eltype(v)(0.5)
    end
end

function average(Um, v, axj::Axis, i, ::Val{false})
    for j in EDGES(axj)
        Um[j] = (v[j,i]+v[j,i-1])*eltype(v)(0.5)
    end
end


# ------------------------- end of the module ------------------------------------
# TESTS


nx,ny,nz,nhalo = 100, 100, 3, 3
axi = Axis(CLOSED, nx, nhalo)
axj = Axis(CLOSED, ny, nhalo)
axk = Axis(CLOSED, nz, nhalo)
axes = AXES(axi,axj,axk)

scheme = UP
maxorder = 6
if false
    du = zeros(size(axes, (EDGES, CENTERS, CENTERS)))
    omegak= zeros(size(axes, (EDGES, EDGES, CENTERS)))
    omegaj= zeros(size(axes, (EDGES, CENTERS, EDGES)))
    v = zeros(size(axes, (CENTERS, EDGES, CENTERS)))
    w = zeros(size(axes, (CENTERS, CENTERS, EDGES)))

    @. v=1
    @. w=1
    omegaj[:] = 1:length(omegaj)
    omegak[:] = 1:length(omegak)

    T = eltype(du)
    uvrecipe = VFrecipe(T, axes,(1,2,3),scheme,maxorder)
    uwrecipe = VFrecipe(T, axes,(1,3,2),scheme,maxorder,true)

    #rec= VFrecipe(axes,(1,3,2),scheme,maxorder)

    #vortex_force_oneterm(du, v, omegak, uvrecipe)
    vortex_force_oneterm(du, w, omegaj, uwrecipe)
end

U = zeros(size(axes, (EDGES, CENTERS, CENTERS)))
V = zeros(size(axes, (CENTERS, EDGES, CENTERS)))
q = zeros(size(axes, (CENTERS, CENTERS, CENTERS)))
dq = zeros(size(axes, (CENTERS, CENTERS, CENTERS)))

@. U=1
@. V=1
q[:] = 1:length(q)

T = eltype(q)

urecipe = DFrecipe(T, axes, (1,2,3),scheme,maxorder)
divflux_oneterm(dq,U,q,urecipe)

vrecipe = DFrecipe(T, axes, (2,1,3),scheme,maxorder)
divflux_oneterm(dq,V,q,vrecipe)

# function loop_over_views(q)
#     nx,ny,nz = size(q)
#     for i in 1:nx, j in 1:ny
#         z = view1d(q,i,j,Val(3))
#     end
# end

iprecipe = IPrecipe(Float64, axes, (1,2,3), WZ, 6, BYONE)
innerproduct_oneterm(q,U,U,iprecipe)

ke(q::A,U::A,V::A) where {T,A<:Array{T,3}} = begin
    nx,ny,nz = size(q)
    for i in 1:nx, j in 1:ny, k in 1:nz
        q[i,j,k] = U[i,j,k]*U[i,j,k] + U[i+1,j,k]*U[i+1,j,k]
    end
end
