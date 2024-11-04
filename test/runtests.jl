using NetCDF: ncread

using ManagedLoops: @unroll
using LoopManagers: PlainCPU
using MutatingOrNot: void
using ClimFlowsData: DYNAMICO_reader
using CFDomains: VoronoiSphere
using CFDomains.Stencils
using CFTransport
using CFTransport.VoronoiSLFV:  backwards_trajectories!, VoronoiSLFV.SLFV_flux!


using Test

bell((x,y,z)) = max(0, 1-2(z^2+x^2))

function test_SLFV(sphere, mgr)
    # time stepping
    time_end, time_steps = 2pi/10, 50
    dt = time_end / time_steps
    t = zero(dt)
    # stream function for solid-body rotation
    psi = [z for (x,y,z) in sphere.xyz_v]
    # time-integrated contravariant mass flux, exactly non-divergent
    mflux = [dt*Stencils.gradperp(sphere, edge)(psi) for edge in eachindex(sphere.xyz_e)]
    # mass
    mass = one.(sphere.lon_i)
    qmass = bell.(sphere.xyz_i)
    # ensure that extrema(qmass) == (0,1)
    qmass = qmass .- minimum(qmass)
    qmass = qmass/maximum(qmass)

    disp, dx = backwards_trajectories!(void, void, mgr, sphere, mass, mflux)
    qflux, gradq, q = SLFV_flux!(void, void, void, mgr, sphere, disp, mass, mflux, qmass)

    @info "extrema(q)"

    for step in 1:time_steps
        backwards_trajectories!(disp, dx, mgr, sphere, mass, mflux)
        SLFV_flux!(qflux, gradq, q, mgr, sphere, disp, mass, mflux, qmass)

        mod(step,10)==1 && @info extrema(q)

        for (cell, deg) in enumerate(sphere.primal_deg)
            qmass[cell] -= @unroll deg in 5:7 Stencils.divergence(sphere, cell, Val(deg))(qflux)
        end

    end
    @test minimum(q)>=0
    @test maximum(q)<=1
end

@testset "CFTransport.jl" begin
    sphere = VoronoiSphere(DYNAMICO_reader(ncread, "uni.1deg.mesh.nc") ; prec=Float64)
    test_SLFV(sphere, PlainCPU())
end
