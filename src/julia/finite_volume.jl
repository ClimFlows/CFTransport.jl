"""
    abstract type OneDimFV{kind,dim,rank} end

One-dimensional finite volume transport operator acting on dimension `dim` of arrays of rank `rank`.
If `kind==:density` the operator transports a density field (e.g. in kg/m3)
If `kind==:scalar` the operator transports a scalar field (e.g. in kg/kg)
"""
abstract type OneDimFV{kind,Dim,Rank} <: OneDimOp{Dim,Rank} end

# compute concentrations from densities
@loops function concentrations!(_, q, mq, m)
    let range = eachindex(mq)
        @fast @simd for i in range
            q[i] = mq[i]*inv(m[i])
        end
    end
end

function FV_update!(backend, fv::OneDimFV{:density}, newmq_, mq_, fluxq_)
    invoke(fv, backend, axes(fluxq), crop(0,1), (newmq_, mq_, fluxq_)
    ) do i, ::Any, (newmq, mq, fluxq)
        @fast newmq[i] = mq[i] + (fluxq[i]-fluxq[i+1])
    end
end

function FV_update!(backend, fv::OneDimFV{:scalar}, newq_, q_, fluxq_, flux_, m_)
    invoke(fv, backend, axes(fluxq_), crop(0,1), (newq_, q_, fluxq_, flux_, m_)
    ) do i, ::Any, (newq, q, fluxq, flux, m)
        @fast begin
            mq      = m[i]*q[i] + (fluxq[i]-fluxq[i+1])
            newm    = m[i]      + (flux[i]  -flux[i+1])
            newq[i] = mq*inv(newm)
        end
    end
end
