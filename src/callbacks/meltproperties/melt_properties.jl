# function to advect at each timestep
function viscosity_call_func(u, t, integrator)

    @unpack grid, domain, parameters = integrator.p
    @unpack visco_f, compo_f, compo_f_mol, ρ_f = domain

    @inbounds for I in CartesianIndices(visco_f)
      visco_f[I] = melt_viscosity_i(compo_f[I[1], I[2], :], compo_f_mol)
    #   ρ_f[I] = melt_density_i(compo_f[I[1], I[2], :], compo_f_mol)
    end

end

