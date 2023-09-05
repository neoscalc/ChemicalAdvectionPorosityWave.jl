using Parameters

function fluid_flux_ad!(q_f, ϕ, Pe, v_s, Properties, Domain, Grid)
    @unpack nz, nx = Grid
    @unpack n, Δz_ad, Δx_ad, visco_f0, ρ0 = Properties
    @unpack visco_f, ρ_s, ρ_f, g = Domain

    Base.@propagate_inbounds @inline av_k_z(i,in,j) = 0.5*(ϕ[i,j]^n+ϕ[in,j]^n)
    Base.@propagate_inbounds @inline av_k_x(i,j,je) = 0.5*(ϕ[i,j]^n+ϕ[i,je]^n)
    Base.@propagate_inbounds @inline av_μf_z(i,in,j) = 0.5*(visco_f[i,j]+visco_f[in,j]) / visco_f0
    Base.@propagate_inbounds @inline av_μf_x(i,j,je) = 0.5*(visco_f[i,j]+visco_f[i,je]) / visco_f0
    Base.@propagate_inbounds @inline av_Δρ_z(i,in,j) = (ρ_s - 0.5*(ρ_f[i,j] + ρ_f[in,j])) * g / (ρ0*g)

    for I in CartesianIndices(q_f[:z])
        i,j = Tuple(I)
        is, in = limit_periodic(i-1, nz), limit_periodic(i, nz)

        q_f[:z][I] = av_k_z(is,in,j) / av_μf_z(is,in,j) * ((Pe[in,j]-Pe[is,j]) / Δz_ad + av_Δρ_z(is,in,j)) + v_s[:z][I] * 0.5 * (ϕ[in,j] + ϕ[is,j])
    end

    for I in CartesianIndices(q_f[:x])
        i,j = Tuple(I)
        jw, je = limit_periodic(j-1, nx), limit_periodic(j, nx)

        q_f[:x][I] = av_k_x(i,jw,je) / av_μf_x(i,jw,je) * (Pe[i,je]-Pe[i,jw]) / Δx_ad + v_s[:x][I] * 0.5 * (ϕ[i,je] + ϕ[i,jw])
    end
end

function fluid_velocity_ad!(v_f, q_f, ϕ, Grid)

    @unpack nz, nx = Grid

    for I in CartesianIndices(v_f[:z])
        i,j = Tuple(I)
        is, in = limit_periodic(i-1, nz), limit_periodic(i, nz)

        v_f[:z][I] = q_f[:z][I] / ((ϕ[is,j] + ϕ[in,j]) / 2)
    end

    for I in CartesianIndices(v_f[:x])
        i,j = Tuple(I)
        jw, je = limit_periodic(j-1, nx), limit_periodic(j, nx)

        v_f[:x][I] = q_f[:x][I] / ((ϕ[i,jw] + ϕ[i,je]) / 2)
    end

end

function solid_velocity_ad!(vc_s, ϕ, ϕ_prev, Δt, Properties, Domain, Grid)
    @unpack nz, nx = Grid
    @unpack n, Δz_ad, Δx_ad, visco_f0, ρ0 = Properties
    @unpack visco_f, ρ_s, ρ_f, g = Domain

    for I in CartesianIndices(vc_s[:z])
        vc_s[:z][I] = (ϕ[I] - ϕ_prev[I]) / Δt * Δz_ad
    end

end

function velocity_to_center!(vc, v)

    # take velocity on the staggered grid and take the mean to cell centers

    for I = CartesianIndices(vc[:x])
        i, j = Tuple(I)

        vc[:z][I] = 0.5 * (v[:z][i,j] + v[:z][i+1,j])
        vc[:x][I] = 0.5 * (v[:x][i,j] + v[:x][i,j+1])
    end

end


function velocity_to_sides!(v, vc, Grid)

    @unpack nz, nx = Grid

    for I = CartesianIndices(v[:z])
        i, j = Tuple(I)
        is, in = limit_periodic(i-1, nz), limit_periodic(i, nz)

        v[:z][I] = 0.5 * (vc[:z][is,j] + vc[:z][in,j])
    end

    for I = CartesianIndices(v[:x])
        i, j = Tuple(I)
        jw, je = limit_periodic(j-1, nx), limit_periodic(j, nx)

        v[:x][I] = 0.5 * (vc[:x][i,jw] + vc[:x][i,je])
    end


end

# function to advect at each timestep
function velocity_call_func(u, t, integrator)

    @unpack grid, domain, parameters = integrator.p
    @unpack vc_s, v_s, q_f, v_f = parameters

    Δt_ad = integrator.t - integrator.tprev
    ϕ = @view integrator.u[:,:,2]
    ϕ_prev = @view integrator.uprev[:,:,2]
    Pe = @view integrator.u[:,:,1]
    Pe_prev = @view integrator.uprev[:,:,1]

    solid_velocity_ad!(vc_s, ϕ, ϕ_prev, Δt_ad, parameters, domain, grid)
    velocity_to_sides!(v_s, vc_s, grid)
    fluid_flux_ad!(q_f, ϕ, Pe, v_s, parameters, domain, grid)
    fluid_velocity_ad!(v_f, q_f, ϕ, grid)
end
