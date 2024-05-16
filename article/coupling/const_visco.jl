using ChemicalAdvectionPorosityWave

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
# SiO2, TiO2, Al2O3, FeO, MgO, CaO, Na2O, K2O, H2O
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
# SiO2, TiO2, Al2O3, FeO, MgO, CaO, Na2O, K2O, H2O
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
algo_name = "WENO"

# Courant number has to be lower than 1 for upwind and WENO-5, can be higher for MIC and QMSL
Courant_nb = 0.7

# path to save the output data
path_hdf5 = joinpath(@__DIR__,"output_$(grid.nx)x$(grid.nz)_$(algo_name)_C_$(Courant_nb)_1.5Myr_test.h5")

@show grid.nx, grid.nz
@show algo_name

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
output_call = PresetTimeCallback(save_time,save_data, save_positions=(false,false))

# define a steplimiter for the timestep of the two-phase flow (depends on the courant number of the magma)
steplimiter = StepsizeLimiter(dtmaxC;safety_factor=10//10,max_step=false,cached_dtcache=0.0)

# define the order in which to call the callback functions
callbacks = CallbackSet(velocity_call, advection_call, steplimiter, plotting, output_call)

simulate(model; callbacks=callbacks)