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
    invoke(fv, backend, axes(fluxq_), crop(0,1), (newmq_, mq_, fluxq_)
    ) do (i,j), ::Any, (newmq, mq, fluxq)
        @fast newmq[i,j] = mq[i,j] + (fluxq[i,j]-fluxq[i+1,j])
    end
end

function FV_update!(backend, fv::OneDimFV{:scalar}, newq_, q_, fluxq_, flux_, m_)
    invoke(fv, backend, axes(fluxq_), crop(0,1), (newq_, q_, fluxq_, flux_, m_)
    ) do (i,j), ::Any, (newq, q, fluxq, flux, m)
        @fast begin
            mq      = m[i,j]*q[i,j] + (fluxq[i,j]-fluxq[i+1,j])
            newm    = m[i,j]      + (flux[i,j]  -flux[i+1,j])
            newq[i,j] = mq*inv(newm)
        end
    end
end
