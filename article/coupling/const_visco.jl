using ChemicalAdvectionPorosityWave


# path to save the output data
path_hdf5 = joinpath(@__DIR__,"output.h5")


# define the resolution of the grid and the size of the model
grid = Grid(nx=100, nz=200, Lx=450.0u"m", Lz=900.0u"m", tfinal=1.5u"Myr")

domain = Domain(x=grid.x, z=grid.z, nb_element=9)

# gaussian anomaly as initial conditions
a = 0.05 # max porosity in the gaussian
bx = grid.Lx÷2 # x position of the gaussian in m
bz = grid.Lz÷6 # z position of the gaussian in m
σ = 30 # standard deviation of the gaussian

# define initial porosity as a gaussian
domain.ϕ .= 1e-3 .+ a .* exp.(-((grid.x' .* ones(size(grid.z)) .- bx).^2 .+ (ones(size(grid.x))' .* grid.z .- bz).^2) ./ (σ)^2)

# ETN, Basalt D. Giordano and D.B. Dingwell, 2003 renormalised to 100 wt%
# SiO2, TiO2, Al2O3, FeO, MgO, CaO, Na2O, K2O
compo_basalt = [48.32,
                1.65,
                16.72,
                10.41,
                5.31,
                10.75,
                3.85,
                1.99,
                1.00
                ]


# Andesite, Neuville et al, 1992 renormalised to 100 wt%
# SiO2, TiO2, Al2O3, FeO, MgO, CaO, Na2O, K2O
compo_andesit = [59.87,
                 0.82,
                 16.93,
                 5.28,
                 3.28,
                 5.70,
                 3.76,
                 1.36,
                 3
                 ]

x0 = grid.Lx÷2  # x position of the gaussian in m
z0 = grid.Lz÷6  # z position of the gaussian in m
R = 60  # radius of the cylinder in m

# define centered circle anomaly as initial conditions for the magma composition. Basalt in the circle and andesite outside the circle.
for I in CartesianIndices(domain.compo_f[:,:,1])
    r = sqrt((grid.grid[1][I] - x0)^2 + (grid.grid[2][I] - z0)^2)
    if r <= R
        for k in 1:size(domain.compo_f, 3)
            domain.compo_f[I[1], I[2], k] = compo_andesit[k]
        end
    else
        for k in 1:size(domain.compo_f, 3)
            domain.compo_f[I[1], I[2], k] = compo_basalt[k]
        end
    end
end

# to choose the advection schemes, 4 options: UW (upwind), WENO (WENO-5), SL (quasi-monotone semi-Lagrangian) and MIC (marker-in-cell)
algo_name = "MIC"

# Courant number has to be lower than 1 for upwind and WENO-5, can be higher for MIC and QMSL
Courant_nb = 0.7

# define structures for the models
advection_algo = advection(;algo_name=algo_name, grid=grid, compo_f=domain.compo_f)
model = Model(grid=grid, domain=domain, advection_algo=advection_algo, path_data=path_hdf5, Courant=Courant_nb)

# define callbacks to call at the end of each timestep

# compute flux and velocity fields for the solid and fluid phase
velocity_call = FunctionCallingCallback(velocity_call_func; funcat=Vector{Float64}(), func_everystep=true, func_start = false, tdir=1);

# call advection scheme and advect the chemical composition of the magma
advection_call = FunctionCallingCallback(advection_call_func; funcat=Vector{Float64}(), func_everystep=true, func_start = false, tdir=1);

# define plotting function
plotting = FunctionCallingCallback(plotting_tpf; funcat=Vector{Float64}(), func_everystep=true, func_start = false, tdir=1);

# define saving output at 4 time -> 0.35 Myr, 0.80 Myr, 1.20 Myr and at the final timestep
@unpack compaction_t = model
save_time = [ustrip(u"s", 0.35u"Myr") / compaction_t , ustrip(u"s", 0.80u"Myr") / compaction_t, ustrip(u"s", 1.20u"Myr") / compaction_t, grid.tfinal / compaction_t]
output_call = PresetTimeCallback(save_time,save_data)

# define a steplimiter for the timestep of the two-phase flow (depends on the courant number of the magma)
steplimiter = StepsizeLimiter(dtmaxC;safety_factor=10//10,max_step=false,cached_dtcache=0.0)

# define the order in which to call the callback functions
callbacks = CallbackSet(velocity_call, advection_call, steplimiter, plotting, output_call)
# callbacks = CallbackSet(velocity_call, advection_call, steplimiter, output_call)


@unpack grid, domain, advection_algo, ϕ_ini, u0, du0 = model
@unpack compaction_ϕ, compaction_t, compaction_l, compaction_Pe = model

# save output initial conditions if a path is defined in a hdf5 file
if model.path_data !== ""
    hdf5_start(grid, domain, model.path_data);
    hdf5_initial_conditions(grid, domain, model)
end

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

