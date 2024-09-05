#============== Remapping fluxes ============#

"""
    remap_fluxes!(flux, newmass, mass, layout, vcoord::PressureCoordinate)

Computes the target (pseudo-)mass distribution `newmass`
prescribed by pressure-based coordinate `vcoord` and surface pressure `ps`,
as well as the vertical (pseudo-)mass flux `flux` needed to remap
from current mass distribution `mass` to target `newmass`.
`layout` specifies the data layout, see `VHLayout` and `HVLayout`.
"""
function remap_fluxes! end

# convention:
#      fun(non-fields, output fields..., #==# scratch #==# input fields...)
# or   fun(non-fields, output fields..., #==# input fields...)


const AA{Rank, T} = AbstractArray{T, Rank}            # to dispatch on AA{Rank}
const AAV{Rank, T} = Union{Void, AbstractArray{T, Rank}} # AA or Void (output arguments)

function remap_fluxes!(mgr, vcoord::PressureCoordinate, layout::VHLayout, flux::AAV{N}, newmass::AAV{N}, #==# mass::AA{N}) where N
    newmass = alloc_newmass(newmass, layout, mass)
    flux = alloc_flux(flux, layout, mass)
    remap_fluxes_VH!(mgr, vcoord, flux, newmass, mass)
    return flux, newmass
end

function remap_fluxes!(mgr, vcoord::PressureCoordinate, layout::HVLayout{1}, flux::AAV{2}, newmass::AAV{2}, #==# mass::AA{2})
    newmass = alloc_newmass(newmass, layout, mass)
    flux = alloc_flux(flux, layout, mass)
    remap_fluxes_HV!(mgr, vcoord, flux, newmass, mass)
    return flux, newmass
end

function remap_fluxes!(mgr, vcoord::PressureCoordinate, layout::HVLayout{2}, flux::AAV{3}, newmass::AAV{3}, #==# mass::AA{3})
    newmass = alloc_newmass(newmass, layout, mass)
    flux = alloc_flux(flux, layout, mass)
    remap_fluxes_XYV!(mgr, vcoord, flux, newmass, mass)
    return flux, newmass
end

# allocate output argument when it is `void`
alloc_newmass(newmass, layout, _) = newmass
alloc_newmass(::Void, layout, mass) = similar(mass)

alloc_flux(flux, layout, _) = flux

alloc_flux(::Void, layout::VHLayout{1}, mass) =
    similar(mass, size(mass,1)+1, size(mass,2))
alloc_flux(::Void, layout::HVLayout{1}, mass) =
    similar(mass, size(mass,1), size(mass,2)+1)
alloc_flux(::Void, layout::HVLayout{2}, mass) =
    similar(mass, size(mass,1), size(mass,2), size(mass,3)+1)

@loops function remap_fluxes_VH!(_, vcoord, flux, newmass, mass)
    let range = axes(newmass, 2)
        # flux[N+1, nx] (interfaces)
        # mass[N+1, nx]  (levels)
        # ps[nx]  (levels)
        for ij in range
            # total mass in column ij
            masstot = mass[1, ij]
            for k = 2:N
                masstot += mass[k, ij]
            end

            flux[1, ij] = 0
            for k = 1:N
                # integrate vertically the difference between
                # current mass distribution and target mass distribution
                # to get fluxes through interfaces
                newmass[k, ij] = mass_level(2k - 1, masstot, vcoord)
                flux[k+1, ij] = flux[k, ij] + (mass[k, ij] - newmass[k, ij])
            end
        end
    end
end

@loops function remap_fluxes_HV!(_, vcoord, flux, newmass, mass)
    let range = axes(newmass, 1)
        # flux[nx, N+1] (interfaces)
        # mass[nx, N]   (levels)
        # ps[nx]  (levels)
        N = nlayer(vcoord)
        masstot = @view flux[:, N+1] # use flux as buffer
        # total mass
        #=@vec=# for ij in range
            masstot[ij] = mass[ij, 1]
        end
        for k = 2:N
            #=@vec=# for ij in range
                masstot[ij] += mass[ij, k]
            end
        end
        # integrate vertically the difference between
        # current mass distribution and target mass distribution
        # to get fluxes through interfaces
        #=@vec=# for ij in range
            flux[ij, 1] = 0
        end
        for k = 1:N
            #=@vec=# for ij in range
                newmass[ij, k] = mass_level(2k - 1, masstot[ij], vcoord)
                flux[ij, k+1] = flux[ij, k] + (mass[ij, k] - newmass[ij, k])
            end
        end
    end
end

@loops function remap_fluxes_XYV!(_, vcoord, flux, newmass, mass)
    let (irange, jrange) = (axes(newmass, 1), axes(newmass, 2))
        # flux[nx, ny, N+1] (interfaces)
        # mass[nx, ny, N+1]  (levels)
        # ps[nx]  (levels)
        N = nlayer(vcoord)
        masstot = @view flux[:, :, N+1] # use flux as buffer
        # total mass
        #=@vec=# for i in irange, j in jrange
            masstot[i, j] = mass[i, j, 1]
        end
        for k = 2:N
            #=@vec=# for i in irange, j in jrange
                masstot[i, j] += mass[i, j, k]
            end
        end
        # integrate vertically the difference between
        # current mass distribution and target mass distribution
        # to get fluxes through interfaces
        #=@vec=# for i in irange, j in jrange
            flux[i, j, 1] = 0
        end
        for k = 1:N
            #=@vec=# for i in irange, j in jrange
                newmass[i, j, k] = mass_level(2k - 1, masstot[i, j], vcoord)
                flux[i, j, k+1] = flux[i, j, k] + (mass[i, j, k] - newmass[i, j, k])
            end
        end
    end
end
