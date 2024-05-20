using Parameters

function fluid_flux_ad!(qc_f, ϕ, Pe, vc_s, Properties, Domain, Grid)
    @unpack nz, nx = Grid
    @unpack n, Δz_ad, Δx_ad, ρ0 = Properties
    @unpack ρ_s, ρ_f = Domain

    Base.@propagate_inbounds @inline avzPe(i,in,j) = (Pe[i,j]+Pe[in,j]) / 2
    Base.@propagate_inbounds @inline avxPe(i,j,je) = (Pe[i,j]+Pe[i,je]) / 2


    @inbounds @threads for I in CartesianIndices(qc_f[:z])
        i,j = Tuple(I)
        is, in = limit_periodic(i-1, nz), limit_periodic(i+1, nz)
        jw, je = limit_periodic(j-1, nx), limit_periodic(j+1, nx)

        qc_f[:z][I] = ϕ[I]^n * ((avzPe(i,in,j) - avzPe(is,i,j)) / Δz_ad + (ρ_s - ρ_f[i,j]) / (ρ0))

        qc_f[:x][I] = ϕ[I]^n * (avxPe(i,j,je) - avxPe(i,jw,j)) / Δx_ad
    end

end


function fluid_velocity_ad!(vc_f, qc_f, ϕ)
    @inbounds @threads for I in CartesianIndices(vc_f[:z])
        vc_f[:z][I] = qc_f[:z][I] / ϕ[I]
        vc_f[:x][I] = qc_f[:x][I] / ϕ[I]
    end
end


function velocity_to_center!(vc, v)

    # take velocity on the staggered grid and take the mean to cell centers

    @inbounds @threads for I = CartesianIndices(vc[:x])
        i, j = Tuple(I)

        vc[:x][I] = 0.5 * (v[:x][I] + v[:x][i,j+1])  # x
        vc[:z][I] = 0.5 * (v[:z][I] + v[:z][i+1,j])  # z
    end

end

function velocity_to_sides!(v, vc, Grid)

    @unpack nz, nx = Grid

    @inbounds @threads for I = CartesianIndices(v[:z])
        i, j = Tuple(I)
        is, in = limit_periodic(i-1, nz), limit_periodic(i, nz)

        v[:z][I] = 0.5 * (vc[:z][is,j] + vc[:z][in,j])
    end

    @inbounds @threads for I = CartesianIndices(v[:x])
        i, j = Tuple(I)
        jw, je = limit_periodic(j-1, nx), limit_periodic(j, nx)

        v[:x][I] = 0.5 * (vc[:x][i,jw] + vc[:x][i,je])
    end


end

# function to advect at each timestep
function velocity_call_func(u, t, integrator)

    @unpack grid, domain, parameters = integrator.p
    @unpack vc_s, v_s, q_f, v_f, qc_f, vc_f = parameters

    ϕ = @view integrator.u[:,:,2]
    Pe = @view integrator.u[:,:,1]

    fluid_flux_ad!(qc_f, ϕ, Pe, vc_s, parameters, domain, grid)
    fluid_velocity_ad!(vc_f, qc_f, ϕ)
    velocity_to_sides!(v_f, vc_f, grid)
end
