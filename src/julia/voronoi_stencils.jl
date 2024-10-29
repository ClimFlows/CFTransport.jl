module VoronoiStencils

using ManagedLoops: @unroll
using CFDomains.Stencils: Fix, @inl, @gen

# For testing purposes, we temporarily accept an extra `layout` argument
# wich can be `::HVLayout`. This possibility is not part of the API
# and will be removed later. The officially supported layout is VHLayout.
using CFDomains: VHLayout as VH, HVLayout as HV

# official API
@inl limiter(vsphere, cell, v::Val) = limiter(vsphere, VH{1}(), cell, v)

# internal
@gen limiter(vsphere, layout, cell, v::Val{deg}) where {deg} = quote
    neighbours = @unroll (vsphere.primal_neighbour[edge, cell] for edge = 1:$deg)
    shifts = @unroll (vsphere.cen2edge[edge, cell] for edge = 1:$deg)
    Fix(get_limiter, (layout, v, cell, neighbours, shifts))
end

@gen get_limiter(::VH, ::Val{deg}, cell, neighbours, shifts, q, gradq3d, k) where {deg} =
    quote
        # gradq3d is expected to have layout [k, dim, cell] (VDHLayout)
        # q is expected to have layout [k, cell] (VHLayout)
        # all values of q are shifted by qcenter
        # calculate the min and max of q over the current primal cell and its neighbours
        qcenter = q[k, cell]
        dq = @unroll (q[k, neighbours[iedge]] - qcenter for iedge = 1:$deg)
        mini = min(zero(qcenter), minimum(dq))
        maxi = max(zero(qcenter), maximum(dq))

        # we represent alpha as num/den so that at most one division is needed, after the loop
        num, den = one(qcenter), one(qcenter)
        @unroll for iedge = 1:$deg
            dxyz = shifts[iedge]
            edge_est = sum(gradq3d[k, dim, cell] * dxyz[dim] for dim = 1:3)
            edge_est > maxi &&
                num * edge_est > den * maxi &&
                ((num, den) = (maxi, edge_est))
            edge_est < mini &&
                num * edge_est < den * mini &&
                ((num, den) = (mini, edge_est))
        end
        return num < den ? num / den : one(den)
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

        # we represent alpha as num/den so that at most one division is needed, after the loop
        num, den = one(qcenter), one(qcenter)
        @unroll for iedge = 1:$deg
            dxyz = shifts[iedge]
            edge_est = sum(gradq3d[dim, cell, k] * dxyz[dim] for dim = 1:3)
            edge_est > maxi &&
                num * edge_est > den * maxi &&
                ((num, den) = (maxi, edge_est))
            edge_est < mini &&
                num * edge_est < den * mini &&
                ((num, den) = (mini, edge_est))
        end
        return num < den ? num / den : one(den)
    end

end
