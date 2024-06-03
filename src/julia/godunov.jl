"""
    godunov! = GodunovScheme(kind, dim, rank) <: OneDimFV{dim, rank, kind}
    godunov!(backend, newtransported, transported, flux, mass)

Returns the callable `godunov!` that applies a one-dimensional
Godunov scheme to dimension `dim` among `rank`.
Computations are offloaded to `backend`.

If `kind==:density` then `transported` is a density.
If `kind==:scalar` then `transported` is a scalar
whose density is `mass*transported`.

If `newtransported` is the same as `transported`, it is updated in-place.

The Godunov scheme has 2, possibly 3 steps :
0 - (for densities) : compute scalar from density
1 - compute fluxes (upwind/downwind)
2 - update scalar or density

Boundary conditions are left to the user. If BCs must be
enforced between steps 1-2 (e.g. zero boundary fluxes), the user should rather call
individual steps `concentrations!`, `fluxes!` and `FV_update!` .
"""
struct GodunovScheme{kind, Dim, Rank} <: OneDimFV{kind, Dim, Rank}
    GodunovScheme(kind::Symbol, dim=1::Int, rank=1::Int) = new{kind, dim, rank}()
end

# deferring (scheme::GS)(...) to call_godunov(scheme, ...) produces more helpful messages if error
const GS=GodunovScheme
(scheme::GS)(args...) = call_godunov(scheme, args...)

function call_godunov(scheme::GS{:scalar}, backend, newq, q, flux, m)
    fluxq = similar(flux)
    fluxes!(backend, scheme, fluxq, flux, q)
    FV_update!(backend, scheme, newq, q, fluxq, flux, m)
end

function call_godunov!(scheme::GS{:density}, backend, newmq, mq, flux, m)
    q, fluxq = map(similar, (mq, flux))
    concentrations!(backend, q, mq, m)
    fluxes!(backend, scheme, fluxq, flux, q)
    FV_update!(backend, scheme, newmq, mq, fluxq)
end

function fluxes!(backend, scheme::GS, fluxq, flux, q)
    invoke(scheme, backend, axes(fluxq), crop(1,1), (fluxq, flux, q)
    ) do i, scheme, (fluxq, flux, q)
        @fast begin
            flx, qq_up, qq_down = flux[i], q[i-1], q[i]
            fluxq[i] = (1//2) * ( flx*(qq_up+qq_down)+ abs(flx)*(qq_up-qq_down) )
        end
    end
end
