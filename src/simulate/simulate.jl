
function simulate(model; callbacks=false)
    @unpack grid, domain, advection_algo, ϕ_ini, u0, du0 = model
    @unpack compaction_ϕ, compaction_t, compaction_l, compaction_Pe = model

    # save output initial conditions if a path is defined in a hdf5 file
    # if model.path_data !== ""
    #     hdf5_start(grid, domain, model.path_data);
    #     hdf5_initial_conditions(grid, domain, model)
    # end

    #define initial conditions for the model
    u0[:,:,1] .= zeros(grid.nz, grid.nx)
    u0[:,:,2] .= ϕ_ini ./ compaction_ϕ

    p = (grid=grid, domain=domain, parameters=model, advection=advection_algo)
    t = [0, grid.tfinal / compaction_t]  # time scale

    println("Defining Jacobian sparcity...")
    # compute the jacobian sparcity using Symbolics
    jac_sparsity = Symbolics.jacobian_sparsity((du, u)->porosity_wave(du, u, p, 0.0), du0, u0);
    println("Done")

    # define an odefunction with the sparsity of the Jacobian
    f = ODEFunction(porosity_wave;jac_prototype=float.(jac_sparsity));

    println("Defining the problem...")
    # Declare problem with sparcity of the Jacobian
    prob_sparse = ODEProblem(f, u0, t, p);
    println("Done")

    println("Solving the problem...")

    @time sol = solve(prob_sparse, TRBDF2(linsolve=UMFPACKFactorization()), progress = true, callback = callbacks,
    progress_steps = 1, save_everystep=false;)

    println("Done")

    return sol
end