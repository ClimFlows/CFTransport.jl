#=============================== Remapping fluxes ===========================#

@inline remap_fluxes_ps!(backend, flux, newmg, mg, layout::VHLayout, vcoord::PressureCoordinate) =
    remap_fluxes_VH!(backend, flatten(flux, layout), flatten(newmg, layout), flatten(mg, layout), vcoord)

@inline remap_fluxes_ps!(backend, flux, newmg, mg, layout::HVLayout, vcoord::PressureCoordinate) =
    remap_fluxes_HV!(backend, flatten(flux, layout), flatten(newmg, layout), flatten(mg, layout), vcoord)

flatten(x, ::HVLayout{1}) = x
flatten(x, ::VHLayout{1}) = x
flatten(x, ::HVLayout{2}) = reshape(x, :, size(x, 3))
flatten(x, ::VHLayout{2}) = reshape(x, size(x, 1), :)

@loops function remap_fluxes_VH!(_, flux, newmg, mg, vcoord)
    let range = axes(newmg, 2)
        # flux[N+1, nx] (interfaces)
        # mg[N+1, nx]  (levels)
        # ps[nx]  (levels)
        for ij in range
            # total mass in column ij
            masstot = mg[1, ij]
            for k = 2:N
                masstot += mg[k, ij]
            end

            flux[1, ij] = 0
            for k = 1:N
                # integrate vertically the difference between
                # current mass distribution and target mass distribution
                # to get fluxes through interfaces
                newmg[k, ij] = mass_level(2k - 1, masstot, vcoord)
                flux[k+1, ij] = flux[k, ij] + (mg[k, ij] - newmg[k, ij])
            end
        end
    end
end

@loops function remap_fluxes_HV!(_, flux, newmg, mg, vcoord)
    let range = axes(newmg, 1)
        # flux[nx, N+1, nx] (interfaces)
        # mg[nx, N+1]  (levels)
        # ps[nx]  (levels)
        N = levels(vcoord)
        masstot = @view flux[:, N+1] # use flux as buffer
        # total mass
        @vec for ij in range
            masstot[ij] = mg[ij, 1]
        end
        for k = 2:N
            @vec for ij in range
                masstot[ij] += mg[ij, k]
            end
        end
        # integrate vertically the difference between
        # current mass distribution and target mass distribution
        # to get fluxes through interfaces
        @vec for ij in range
            flux[ij, 1] = 0
        end
        for k = 1:N
            @vec for ij in range
                newmg[ij, k] = mass_level(2k - 1, masstot[ij], vcoord)
                flux[ij, k+1] = flux[ij, k] + (mg[ij, k] - newmg[ij, k])
            end
        end
    end
end

