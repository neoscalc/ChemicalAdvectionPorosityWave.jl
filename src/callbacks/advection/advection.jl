using Parameters

function advection(;grid, compo_f, algo_name="SL")

    algo_name = lowercase(algo_name)

    if algo_name == "upwind" || algo_name == "uw"
        UWScheme(nx=grid.nx, nz=grid.nz)
    elseif algo_name == "semi-lagrangian" || algo_name == "sl"
        SemiLagrangianScheme(nx=grid.nx, nz=grid.nz, u0=compo_f)
    elseif algo_name == "weno"
        WENOScheme(nx=grid.nx, nz=grid.nz, u0=compo_f)
    elseif algo_name == "mic"
        MICScheme(nx=grid.nx, nz=grid.nz, Lx=grid.Lx, Lz=grid.Lz, Δx=grid.Δx, Δz=grid.Δz, u0=compo_f)
    else
        error("The name of your algorithm for advection doesn't exist")
    end

end

function advection_call_func(u, t, integrator)

    @unpack grid, domain, parameters, advection = integrator.p
    @unpack compo_f, compo_f_prev = domain
    @unpack compaction_t, compaction_ϕ, compaction_l, v_f = parameters
    @unpack algo_name = advection
    @unpack ϕ_ini = parameters

    ϕ = u[:,:,2] .*  compaction_ϕ
    Δt = (integrator.t - integrator.tprev)

    if algo_name == "Upwind"
        @unpack vc_f = advection
        velocity_to_center!(vc_f, v_f)

        upwind_scheme!(compo_f, compo_f_prev, vc_f, Δt, grid, parameters)
    elseif algo_name == "WENO"
        @unpack vc_f = advection
        velocity_to_center!(vc_f, v_f)

        WENO_scheme!(compo_f, vc_f, advection, grid, parameters, Δt; method=:JS)
    elseif algo_name == "Semi-Lagrangian"
        @unpack vc_previous, vc_f = advection
        velocity_to_center!(vc_f, v_f)

        semi_lagrangian!(compo_f, advection, vc_f, vc_previous, Δt, grid, parameters, ϕ, ϕ_ini; method=:quasi_monotone)

        # save previous velocity
        vc_previous[:x] .= vc_f[:x]
        vc_previous[:z] .= vc_f[:z]

    elseif algo_name == "MIC"
        # initialise markers if first timestep
        if iszero(integrator.tprev)
            MIC_convert_adimensional(advection, compaction_l)

            MIC_initialize_markers!(advection.u_mark, compo_f, advection, parameters)
        end
        @unpack vc_f, density_mark, u_mark, X_mark, Z_mark= advection

        velocity_to_center!(vc_f, v_f)

        MIC!(compo_f, v_f, vc_f, advection, parameters, Δt, grid)

        reseeding_marker!(u_mark, X_mark, Z_mark, density_mark, advection, parameters)
    end

    # # normalize values to 100%
    # sum_element = sum(compo_f[:,:,1:end-1], dims=3)

    # for I = CartesianIndices(sum_element)
    #     i, j = Tuple(I)
    #     # threshold for composition and porosity
    #     for k = 1:(size(domain.compo_f, 3)-1)
    #         compo_f[i,j,k] = compo_f[i,j,k] / sum_element[I] * 100
    #     end
    # end

    # normalize values to 100%
    sum_element = sum(compo_f, dims=3)

    for I = CartesianIndices(sum_element)
        i, j = Tuple(I)
        # threshold for composition and porosity
        for k = axes(compo_f, 3)
            compo_f[i,j,k] = compo_f[i,j,k] / sum_element[I] * 100
        end
    end
end