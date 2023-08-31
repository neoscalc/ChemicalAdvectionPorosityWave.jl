using ChemicalAdvectionPorosityWave

path_hdf5 = joinpath(pwd(), "Example/const_viscosity/output.hdf5")

# physical properties
grid = Grid(nx=100, nz=200, Lx=450u"m", Lz=900u"m", tfinal=1.5u"Myr")

domain = Domain(x=grid.x, z=grid.z, nb_element=9)

# gaussian anomaly as initial conditions
a = 0.05 # max porosity in the gaussian
bx = grid.Lx÷2 # x position of the gaussian in m
bz = grid.Lz÷6 # z position of the gaussian in m
σ = 30 # standard deviation of the gaussian

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

# define centered cylinder anomaly as initial conditions in K
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


advection_algo = advection(;algo_name="MIC", grid=grid, compo_f=domain.compo_f)
model = Model(grid=grid, domain=domain, advection_algo=advection_algo, path_data=path_hdf5, Courant=1.5)


# define callbacks
velocity_call = FunctionCallingCallback(velocity_call_func; funcat=Vector{Float64}(), func_everystep=true, func_start = false, tdir=1);

advection_call = FunctionCallingCallback(advection_call_func; funcat=Vector{Float64}(), func_everystep=true, func_start = false, tdir=1);

plotting = FunctionCallingCallback(plotting_tpf; funcat=Vector{Float64}(), func_everystep=true, func_start = false, tdir=1);

@unpack compaction_t = model
save_time = [ustrip(u"s", 0.35u"Myr") / compaction_t , ustrip(u"s", 0.80u"Myr") / compaction_t, ustrip(u"s", 1.20u"Myr") / compaction_t, grid.tfinal / compaction_t]
output_call = PresetTimeCallback(save_time,save_data)


steplimiter = StepsizeLimiter(dtmaxC;safety_factor=10//10,max_step=false,cached_dtcache=0.0)

callbacks = CallbackSet(velocity_call, advection_call, steplimiter, plotting, output_call)

# run model
callbacks = CallbackSet(velocity_call, advection_call, steplimiter, plotting)
sol = simulate(model, callbacks=callbacks)
