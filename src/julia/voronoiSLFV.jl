module VoronoiSLFV

# slope-limited, semi-Lagrangian finite volume tranport on Voronoi mesh
# Dubey & Dubos, 2015

using CFDomains: VHLayout as VH, HVLayout as HV
using CFDomains.Stencils: Fix, @inl, @gen, average_ie, perp, gradient3d
using ManagedLoops: @with, @unroll, @vec
using MutatingOrNot: similar!

#================== upwind cell =================#

@inl upwind_cell(vsphere, edge) =
    Fix(get_upwind, (vsphere.edge_left_right[1, edge], vsphere.edge_left_right[2, edge]))
@inl get_upwind(left, right, flux, x) = @vec flux > 0 ? x[left] : x[right]

#===============  back-trajectories =============#

"""
    disp, dx = backwards_trajectories!(disp, dx, mgr, vsphere, mass, mflux)

For each edge of `vsphere`, compute the 3D displacement `disp` from the center of the upwind cell
to the midpoint of the backwards trajectory starting from the edge midpoint. 
The backward trajectory is deduced from the `mass` and the time-integrated mass flux `mflux`. 
`dx` is the normal (to the edge) time-integrated velocity.

`disp` and `dx` may be `::Void`, in which case they will be appropriately allocated. 
`vsphere` must be a `VoronoiSphere` or another struct or named tuple with the necessary fields. 
`mgr` is `nothing` or a `LoopManager`. In the latter case, `mgr` manages the computational loops.

`mass` is a scalar field known at primal cells. 
`mflux` is a vector field known by its contravariant components (integrals across primal cell edges). 
`dx` is a vector field known by its normal (to primal cell edges) components. 
`disp` is a 3d-vector-valued field known by its values at edges.

`mass` and `mflux` must be arrays of the same type. `Udt` is similar to `mflux`.
If `mflux` is an AbstractVector representing a 2D field, `disp` is a similar vector, but of 3-uples. 
If `mflux` is an AbstractMatrix representing a 3D field, it must have layout `VHLayout{1}`, i.e.
`nz=size(mflux,1)` is the number of layers. `disp` is then an array of size `(nz, 3, size(mflux,2))`. 
This layout favors SIMD vectorization on CPUs and merged memory accesses on GPUs.
"""
function backwards_trajectories!(
    disp_,
    scratch,
    mgr,
    vsphere,
    mass::AbstractVector,
    mflux::AbstractVector,
)
    # carrier mass and flux => normal component of carrier velocity
    Udt = similar!(scratch, mflux)
    #= @with mgr, =#
    let edges = eachindex(vsphere.xyz_e)
        for edge in edges
            le, avg = vsphere.le[edge], average_ie(vsphere, edge)
            Udt[edge] = mflux[edge] / (avg(mass) * le) # normal component 
        end
    end
    # dx = Udt = normal component => dy = tangential component => disp = displacement
    disp = similar!(disp_, mflux, NTuple{3,eltype(mass)})
    #= @with mgr, =#
    let edges = eachindex(vsphere.xyz_e)
        for edge in edges
            xyz = vsphere.xyz_e[edge] # 3-uples
            norm = vsphere.normal_e[edge] # 3-uples
            tang = vsphere.tangent_e[edge] # 3-uples
            dx, dy = Udt[edge], perp(vsphere, edge)(Udt)
            center = upwind_cell(vsphere, edge)(dx, vsphere.xyz_i)
            disp[edge] = @unroll (
                (xyz[dim] - center[dim]) - (dx * norm[dim] + dy * tang[dim]) / 2 for
                dim = 1:3
            )
        end
    end
    return disp, Udt
end

#===============  SLFV flux =============#

"""
    qflux, gradq, q = SLFV_flux!(qflux, gradq, q, mgr, vsphere, disp, mass, mflux, qmass)

For each edge of `vsphere`, compute the time-integrated scalar flux `qflux` 
following the SLFV scheme, using the displacement `disp` computed by `backwards_trajectories`,
the carrier mass `mass` and time-integrated mass flux `mflux` and the scalar mass `qmass`.
Additionnally, for each primal cell, compute the slope-limited gradient `gradq`,
and mixing ratio `q`

`qflux`, `gradq` and `q` may be `::Void`, in which case they will be appropriately allocated. 
`vsphere` must be a `VoronoiSphere` or another struct or named tuple with the necessary fields. 
`mgr` is `nothing` or a `LoopManager`. In the latter case, `mgr` manages the computational loops.

`mass`, `qmass` and `q` are scalar fields known at primal cells. 
`mflux` and `qflux` are vector fields known by their contravariant components (integrals across primal cell edges). 
`disp` is a 3d-vector-valued field known by its values at edges. 
`gradq` is a 3d-vector-valued field known at primal cells.

If `mflux` is an AbstractVector representing a 2D field, `disp` and `gradq` are vectors of 3-uples. 
If `mflux` is an AbstractMatrix representing a 3D field, it must have layout `VHLayout{1}`, i.e.
`nz=size(mflux,1)` is the number of layers. `disp` and `gradq` are then arrays of size `(nz, 3, size(mflux,2))`
and `(nz, 3, size(qmass, 2))`. 
This layout favors SIMD vectorization on CPUs and merged memory accesses on GPUs.
"""
function SLFV_flux!(
    qflux_,
    gradq_,
    q_,
    mgr,
    vsphere,
    disp::AbstractVector,
    mass::AbstractVector,
    mflux::AbstractVector,
    qmass::AbstractVector,
)
    q = @. mgr[q_] = qmass / mass
    gradq = similar!(gradq_, qmass, NTuple{3,eltype(mass)})
    #= @with mgr, =#
    let cells = eachindex(vsphere.xyz_i)
        for cell in cells
            deg = vsphere.primal_deg[cell]
            @unroll deg in 5:7 begin
                grad = gradient3d(vsphere, cell, Val(deg))(q) # tuple grad = (gx,gy,gz)
                alpha = limiter(vsphere, cell, Val(deg))(q, grad)
                gradq[cell] = @unroll (alpha * grad[dim] for dim = 1:3)
            end
        end
    end
    qflux = similar!(qflux_, mflux)
    #= @with mgr, =#
    let edges = eachindex(vsphere.xyz_e)
        for edge in edges
            flux = mflux[edge]
            up = upwind_cell(vsphere, edge)
            qflux[edge] = flux * (up(flux, q) + dot(up(flux, gradq), disp[edge]))
        end
    end
    return qflux, gradq, q
end

dot((a, b, c)::T, (x, y, z)::T) where {T<:NTuple{3}} = muladd(a, x, muladd(b, y, c * z))

#================ slope limiter =================#

# For testing purposes, we temporarily accept an extra `layout` argument
# wich can be `::HVLayout`. This possibility is not part of the API
# and will be removed later. The officially supported layout is VHLayout.

# official API
@inl limiter(vsphere, cell, v::Val) = limiter(vsphere, VH{1}(), cell, v)

# internal
@gen limiter(vsphere, layout, cell, v::Val{deg}) where {deg} = quote
    neighbours = @unroll (vsphere.primal_neighbour[edge, cell] for edge = 1:$deg)
    shifts = @unroll (vsphere.cen2vertex[edge, cell] for edge = 1:$deg)
    Fix(get_limiter, (layout, v, cell, neighbours, shifts))
end

@gen get_limiter(::VH, ::Val{deg}, cell, neighbours, shifts, q, gradq3d) where {deg} = quote
    # gradq3d is expected to be a tuple (gx,gy,gz)
    # q is expected to have layout [k, cell] (VHLayout)
    # min and max of q-qcenter over the current primal cell and its neighbours
    qcenter = q[cell]
    mini, maxi = zero(qcenter), zero(qcenter)
    @unroll for iedge = 1:$deg
        dq = q[neighbours[iedge]] - qcenter
        mini = min(mini, dq)
        maxi = max(maxi, dq)
    end
    # min and max of linear reconstruction evaluated at cell vertices
    edge_mini, edge_maxi = mini, maxi
    @unroll for iedge = 1:$deg
        dxyz = shifts[iedge]
        edge_est = sum(gradq3d[dim] * dxyz[dim] for dim = 1:3)
        edge_mini = min(edge_est, edge_mini)
        edge_maxi = max(edge_est, edge_maxi)
    end
    # reduce gradient if reconstructed values overshoot
    ratio_mini = (edge_mini < mini) ? mini / edge_mini : one(mini) # >=0
    ratio_maxi = (edge_maxi > maxi) ? maxi / edge_maxi : one(maxi) # >=0
    return min(ratio_mini, ratio_maxi)
end

@gen get_limiter(::VH, ::Val{deg}, cell, neighbours, shifts, q, gradq3d, k) where {deg} =
    quote
        # gradq3d is expected to have layout [k, dim, cell] (VDHLayout)
        # q is expected to have layout [k, cell] (VHLayout)
        # min and max of q-qcenter over the current primal cell and its neighbours
        qcenter = q[k, cell]
        mini, maxi = zero(qcenter), zero(qcenter)
        @unroll for iedge = 1:$deg
            dq = q[k, neighbours[iedge]] - qcenter
            mini = min(mini, dq)
            maxi = max(maxi, dq)
        end
        # min and max of linear reconstruction evaluated at edge midpoints
        edge_mini, edge_maxi = mini, maxi
        @unroll for iedge = 1:$deg
            dxyz = shifts[iedge]
            edge_est = sum(gradq3d[k, dim, cell] * dxyz[dim] for dim = 1:3)
            edge_mini = min(edge_est, edge_mini)
            edge_maxi = max(edge_est, edge_maxi)
        end
        # reduce gradient if reconstructed values overshoot
        ratio_mini = (edge_mini < mini) ? mini / edge_mini : one(mini) # >=0
        ratio_maxi = (edge_maxi > maxi) ? maxi / edge_maxi : one(maxi) # >=0
        return min(ratio_mini, ratio_maxi)
    end

# to be removed later
@gen get_limiter(::HV{1}, ::Val{deg}, cell, neighbours, shifts, q, gradq3d, k) where {deg} =
    quote
        # all values of q are shifted by qcenter
        # calculate the min and max of q over the current primal cell and its neighbours
        qcenter = q[cell, k]
        dq = @unroll (q[neighbours[iedge], k] - qcenter for iedge = 1:$deg)
        mini = min(zero(qcenter), minimum(dq))
        maxi = max(zero(qcenter), maximum(dq))

        alpha = one(qcenter)
        @unroll for iedge = 1:$deg
            dxyz = shifts[iedge]
            edge_est = sum(gradq3d[dim, cell, k] * dxyz[dim] for dim = 1:3)
            if edge_est > maxi
                alpha = min(alpha, maxi / edge_est) # 0 <= maxi < edge_est  ==> alpha <= 1
            elseif edge_est < mini
                alpha = min(alpha, mini / edge_est) # edge_est < mini <= 0  ==> alpha <= 1
            end
        end
        return alpha # ∈ [0,1]
    end

end # module
