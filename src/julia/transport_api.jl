include("transport_blocks.jl")

" divergence of tracer flux in 2D or 3D "
function divflux(dq::A,q::A,U::V,recipes::Tuple{Vararg{<:DFrecipe}}) where {T,N,
                                                                            A<:Array{T,3},
                                                                            V<:NTuple{N,A}}

    for k in 1:N
        divflux_oneterm(dq,U[k],q,recipes[k])
    end

end

" vortex force in genuine 3D:  tendency, velocity and omega are 3-vectors"
function vortex_force(du::V, omega::V, U::V, recipes) where {T,
                                                             A<:Array{T,3},
                                                             V<:Tuple{A,A,A}}
    components = [1,2,3]
    for ax1 in components
        for ax2 in setdiff(components, ax1)
            ax3 = setdiff(components, [ax1, ax2])
            vortex_force_oneterm(du[ax1], U[ax2], omega[ax3], recipes[ax1][ax2])
        end
    end

end

" vortex force in 2D:  tendency, velocity are 2-vectors, omega is a scalar"
function vortex_force(du::V, omega::A, U::V, recipes) where {T,
                                                             A<:Array{T,3},
                                                             V<:Tuple{A,A}}

    vortex_force_oneterm(du[1], U[2], omega, recipes[1])
    vortex_force_oneterm(du[2], U[1], omega, recipes[2])

end

" inner product in 2D or 3D "
function innerprod(ke::A,u::V,U::V,recipes) where {T,N,
                                                   A<:Array{T,3},
                                                   V<:NTuple{N,A}}
    for k in 1:N
        innerproduct_oneterm(ke,u[k],U[k],recipes[k])
    end

end
