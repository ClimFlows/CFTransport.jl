using NetCDF: ncread

using ManagedLoops: @unroll
using LoopManagers: PlainCPU
using MutatingOrNot: void
using ClimFlowsData: DYNAMICO_reader
using CFDomains: VoronoiSphere
using CFDomains.Stencils: divergence, gradperp
using CFTransport
using CFTransport.VoronoiSLFV: backwards_trajectories!, SLFV_flux!, local_minmax,
                               SL, SL_simple, ML

using BenchmarkTools
using Test

bell((x, y, z)) = max(0, 1 - 2(z^2 + x^2))

function test_SLFV(lim, sphere, mgr, clip)
    @info "Testing with $(typeof(lim)) and $clip:"
    # time stepping
    dt, time_steps = 0.01, 100
    # stream function for solid-body rotation
    psi = [z for (x, y, z) in sphere.xyz_v]
    # time-integrated contravariant mass flux, exactly non-divergent
    mflux = [dt * gradperp(sphere, edge)(psi) for edge in eachindex(sphere.xyz_e)]
    # mass
    mass = one.(sphere.lon_i)
    qmass = bell.(sphere.xyz_i)
    qmass2 = similar(qmass) # for Heun scheme

    # ensure that extrema(qmass) == (0,1)
    qmass = qmass .- minimum(qmass)
    qmass = qmass / maximum(qmass)

    disp, dx = backwards_trajectories!(void, void, mgr, lim, sphere, mass, mflux)
    qflux, gradq, q = SLFV_flux!(void, void, void, mgr, lim, sphere, disp, mass, mflux,
                                 qmass)

    @info "min(q), 1 - max(q), total mass"

    for step in 1:time_steps
        mod(step-1, time_steps / 10) == 0 && @info (minimum(q), 1 - maximum(q), sum(qmass.*sphere.Ai))
        step!(lim, qflux, gradq, q, mgr, sphere, disp, dx, mass, mflux, qmass, qmass2, clip)
    end
    @test minimum(q) + eps(eltype(qmass)) >= 0
    @test maximum(q) <= 1
    bench = @benchmark step!($lim, $qflux, $gradq, $q, $mgr, $sphere, $disp, $dx, $mass, $mflux, $qmass, $qmass2, $clip) seconds=1
    @test bench.allocs == 0
    display(bench)
end

# Euler step
function step!(lim::Union{SL,SL_simple}, qflux, gradq, q, mgr, sphere, disp, dx, mass,
               mflux, qmass, qmass2, clip)
#    backwards_trajectories!(disp, dx, mgr, lim, sphere, mass, mflux)
    SLFV_flux!(qflux, gradq, q, mgr, lim, sphere, disp, mass, mflux, qmass)
    @. qmass2 = qmass
    @inbounds for (cell, deg) in enumerate(sphere.primal_deg)
        @unroll deg in 5:7 begin
            qc, qmin, qmax = local_minmax(sphere, cell, Val(deg))(qmass2)
            qmass[cell] = clip(qmin, qmax, qc - divergence(sphere, cell, Val(deg))(qflux))
        end
    end
end
clip(qmin, qmax, q) = max(qmin, min(qmax, q))
noclip(qmin, qmax, q) = q

# clip(qmin, qmax, q) = q

# Heun step
function step!(lim::ML, qflux, gradq, q, mgr, sphere, disp, dx, mass, mflux, qmass, qmass2, clip)
    SLFV_flux!(qflux, gradq, q, mgr, lim, sphere, disp, mass, mflux, qmass)
    for (cell, deg) in enumerate(sphere.primal_deg)
        divflux = @unroll deg in 5:7 divergence(sphere, cell, Val(deg))(qflux)
        qmass2[cell] = qmass[cell] - divflux
    end
    SLFV_flux!(qflux, gradq, q, mgr, lim, sphere, disp, mass, mflux, qmass2)
    for (cell, deg) in enumerate(sphere.primal_deg)
        divflux = @unroll deg in 5:7 divergence(sphere, cell, Val(deg))(qflux)
        qmass[cell] = (qmass[cell] + (qmass2[cell] - divflux)) / 2
    end
end

norm2((x, y, z)) = x * x + y * y + z * z

function named_tuple(x)
    return NamedTuple{propertynames(x)}(map(sym -> getfield(x, sym), propertynames(x)))
end

@testset "CFTransport.jl" begin
    sphere = VoronoiSphere(DYNAMICO_reader(ncread, "uni.1deg.mesh.nc"); prec=Float64)
    radius2_i = [maximum(norm2,
                         sphere.cen2vertex[edge, cell] for edge in 1:sphere.primal_deg[cell])
                 for cell in eachindex(sphere.Ai)]
    sphere = merge(named_tuple(sphere), (; radius2_i))
    test_SLFV(ML(), sphere, PlainCPU(), noclip)
    test_SLFV(SL(), sphere, PlainCPU(), clip)
    test_SLFV(SL_simple(), sphere, PlainCPU(), clip)
    test_SLFV(SL_simple(), sphere, PlainCPU(), noclip)
end
