"""
    vanleer! = VanLeerScheme(kind, limiter, dim, rank) <: OneDimFV{dim, rank, kind}
    vanleer!(backend, newtransported, transported, flux, mass)

Returns the callable `vanleer` that applies a one-dimensional
Van Leer scheme with `limiter` to dimension `dim` among `rank`.
Computations are offloaded to `backend`.

If `kind==:density` then `transported` is a density.
If `kind==:scalar` then `transported` is a scalar
whose density is `mass*transported`.

If `newtransported` is the same as `transported`, it is updated in-place.

The VanLeer scheme has 3, possibly 4 steps :
0 - (for densities) : compute scalar from density
1 - compute slopes (with slope limiter)
2 - compute fluxes (upwind/downwind)
3 - update scalar or density

Boundary conditions are left to the user. If BCs must be
enforced between steps 1-2 (e.g. zero boundary slopes)
and 2-3 (e.g. zero boundary fluxes), the user should rather call
individual steps `concentrations!`, `slopes!`, `fluxes!` and `FV_update!` .
"""
struct VanLeerScheme{kind, Limiter, Dim, Rank} <: OneDimFV{kind, Dim, Rank}
    limiter :: Limiter
    VanLeerScheme(kind::Symbol, lim::Lim, dim=1::Int, rank=1::Int) where {Lim} = new{kind, Lim, dim, rank}(lim)
end

# deferring (vl::VLS)(...) to call_vls(vl, ...) produces more helpful messages if error
const VLS=VanLeerScheme
@inline (vl::VLS)(args...) = @inline call_vls(vl, args...)

function call_vls(vl::VLS{:scalar}, backend, newq, q, flux, m)
    dq = similar(q)
    fluxq = similar(flux)
    slopes!(backend, vl, dq, q)
    fluxes!(backend, vl, fluxq, m, flux, q, dq)
    update!(backend, vl, newq, q, fluxq, flux, m)
end

function call_vls(vl::VLS{:density}, backend, newmq, mq, flux, m)
    q, dq, fluxq = map(similar, (mq, mq, flux))
    concentrations!(backend, q, mq, m)
    slopes!(backend, vl, dq, q)
    fluxes!(backend, vl, fluxq, m, flux, q, dq)
    update!(backend, vl, newmq, mq, fluxq)
end

function slopes!(backend, vl::VLS, dq, q)
    invoke(vl, backend, axes(dq), crop(1,1), (dq,q) ) do i, vl, (dq,q)
        @fast dq[i] = limited_slope(i, q, vl.limiter)
    end
end

function fluxes!(backend, vl::VLS, fluxq, mass, flux, q, dq)
    invoke(vl, backend, axes(fluxq), crop(1,1), (fluxq, mass, flux, q, dq)
    ) do i, vl, (fluxq, m, flux, q, dq)
        @fastmath @fast begin
            flx = flux[i]
            # upward transport, upwind side is at lower level
            qq_up   = q[i-1] + half(1-flx*inv(m[i-1]))*dq[i-1]
            # downward transport, upwind side is at upper level
            qq_down = q[i] - half(1+flx*inv(m[i]))*dq[i]
            # select upward or downward without branching
            fluxq[i] = half( flx*(qq_up+qq_down)+ abs(flx)*(qq_up-qq_down) )
        end
    end
end
