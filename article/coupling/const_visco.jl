using ChemicalAdvectionPorosityWave

# define the resolution of the grid and the size of the model
# grid = Grid(nx=200, nz=400, Lx=450.0u"m", Lz=900.0u"m", tfinal=1.5u"Myr")
grid = Grid(nx=100, nz=200, Lx=450.0u"m", Lz=900.0u"m", tfinal=1.5u"Myr")

domain = Domain(x=grid.x, z=grid.z, nb_element=10)

# gaussian anomaly as initial conditions
a = 0.05 # max porosity in the gaussian
bx = grid.Lx÷2 # x position of the gaussian in m
bz = grid.Lz÷6 # z position of the gaussian in m
σ = 30 # standard deviation of the gaussian

# define initial porosity as a gaussian
domain.ϕ .= 1e-3 .+ a .* exp.(-((grid.x' .* ones(size(grid.z)) .- bx).^2 .+ (ones(size(grid.x))' .* grid.z .- bz).^2) ./ (σ)^2)

# ETN, Basalt D. Giordano and D.B. Dingwell, 2003 renormalised to 100 wt%
# SiO2, TiO2, Al2O3, FeO, MgO, CaO, Na2O, K2O, H2O, tracer
compo_basalt = [48.32,
                1.65,
                16.72,
                10.41,
                5.31,
                10.75,
                3.85,
                1.99,
                1.00,
                50.0
                ]


# Andesite, Neuville et al, 1992 renormalised to 100 wt%
# SiO2, TiO2, Al2O3, FeO, MgO, CaO, Na2O, K2O, H2O, tracer
compo_andesit = [59.87,
                 0.82,
                 16.93,
                 5.28,
                 3.28,
                 5.70,
                 3.76,
                 1.36,
                 3,
                 100.0
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

# path to save the output data
path_hdf5 = joinpath(@__DIR__,"output_$(grid.nx)x$(grid.nz)_$(algo_name)_C_$(Courant_nb)_1.5Myr.h5")

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
# jac_sparsity = Symbolics.jacobian_sparsity((du, u)->porosity_wave(du, u, p, 0.0), du0, u0);
# println("Done")

# define an odefunction with the sparsity of the Jacobian
f = ODEFunction(porosity_wave;jac_prototype=float.(jac_sparsity));

println("Defining the problem...")
# Declare problem with sparcity of the Jacobian
prob_sparse = ODEProblem(f, u0, t, p);
println("Done")

println("Solving the problem...")

@time sol = solve(prob_sparse, TRBDF2(linsolve=UMFPACKFactorization()), progress = true, callback = callbacks,
progress_steps = 1, save_everystep=false;)

# # MWE to test reseeding

# @unpack compo_f, compo_f_prev = domain
# @unpack compaction_t, compaction_ϕ, compaction_l, v_f, xs_ad, zs_ad, xs_ad_vec, zs_ad_vec = model
# @unpack algo_name, u_mark, X_mark, Z_mark, density_mark, nx, nz, nx_marker, nz_marker, Lz, Lx, nx, nz, nx_marker, nz_marker, X_mark_save, Z_mark_save, vc_t_old, vs_t_old, v_t_old, vc_timestep, vs_timestep, v_timestep, norm_mark, XZ_mark_cell, mark_per_cell, mark_per_cell_array, mark_per_cell_array_el, u0, XZ_mark_sort = advection_algo
# @unpack ϕ_ini = model

# ChemicalAdvectionPorosityWave.MIC_convert_adimensional(advection_algo, compaction_l)

# ChemicalAdvectionPorosityWave.MIC_initialize_markers!(advection_algo.u_mark, compo_f, advection_algo, model)

# X_mark_sort = sort(X_mark)
# Z_mark_sort = sort(Z_mark)
# # @btime ChemicalAdvectionPorosityWave.density_marker_per_cell!($density_mark, $X_mark, $Z_mark, $xs_ad_vec, $zs_ad_vec, $XZ_mark_sort)
# ChemicalAdvectionPorosityWave.density_marker_per_cell!(density_mark, X_mark, Z_mark, xs_ad_vec, zs_ad_vec, XZ_mark_sort)
# density_mark[100, 50:60] .= 0


# # @time ChemicalAdvectionPorosityWave.add_marker_per_cell!(u_mark, X_mark, Z_mark, density_mark, xs_ad, zs_ad, advection_algo)

# ChemicalAdvectionPorosityWave.add_marker_per_cell!(u_mark, X_mark, Z_mark, density_mark, xs_ad, zs_ad, advection_algo)
# @btime ChemicalAdvectionPorosityWave.add_marker_per_cell!($u_mark, $X_mark, $Z_mark, $density_mark, $xs_ad, $zs_ad, $advection_algo) setup = (u_mark=[0], X_mark=[0], Z_mark=[0], density_mark=[0], xs_ad=[0], zs_ad=[0], advection_algo=[0]) evals = 1
# # 7.783 s (76 allocations: 43.09 KiB)
# # 67.498 s (1737 allocations: 338.36 KiB)
# # 64.999 s (482 allocations: 341.23 KiB)

# # 1.244 s (13 allocations: 4.47 KiB)
# # 1.297 s (37 allocations: 7.11 KiB)ca
# ChemicalAdvectionPorosityWave.remove_marker_per_cell!(u_mark, X_mark, Z_mark, density_mark, xs_ad, zs_ad, advection_algo)



# function remove_marker_per_cell!(u_mark, X_mark, Z_mark, density_mark, x, z, MIC)

#     @unpack nx, nz, nx_marker, nz_marker, X_mark_save, Z_mark_save, vc_t_old, vs_t_old, v_t_old, vc_timestep, vs_timestep, v_timestep, norm_mark, u0, X_mark_sort, Z_mark_sort, XZ_mark_sort = MIC

#     @inbounds for I in CartesianIndices(density_mark)
#         i, j = Tuple(I)
#         if density_mark[i, j] > round(THRESHOLD_FACTOR_DENSITY_REMOVE*nx_marker*nz_marker / (nx*nz))

#             nb_el = size(u0, 3)

#             # delete all previous markers inside this cell
#             index_delete::Vector{Int64} = []

#             @inbounds for k in axes(X_mark, 1)
#                 if Z_mark[k] >= z[i] && Z_mark[k] <= z[i+1] && X_mark[k] >= x[j] && X_mark[k] <= x[j+1]
#                     push!(index_delete, k)
#                 end
#             end

#             # keep only one third of the element in index_delete but randomly

#             index_delete = index_delete[1:round(Int, length(index_delete)/3)]

#             # choose half of the markers in the cell
#             marker_to_remove = round(Int, FRACTION_REMOVE_MARKER*length(index_delete))
#             index_delete_choosen::Vector{Int64} = sort(sample(index_delete, marker_to_remove; replace=false))

#             deleteat!(X_mark, index_delete_choosen)
#             deleteat!(Z_mark, index_delete_choosen)
#             deleteat!(Z_mark_save, index_delete_choosen)
#             deleteat!(X_mark_save, index_delete_choosen)
#             deleteat!(vc_t_old[1], index_delete_choosen)
#             deleteat!(vs_t_old[1], index_delete_choosen)
#             deleteat!(v_t_old[1], index_delete_choosen)
#             deleteat!(vc_timestep[1], index_delete_choosen)
#             deleteat!(vs_timestep[1], index_delete_choosen)
#             deleteat!(v_timestep[1], index_delete_choosen)
#             deleteat!(vc_t_old[2], index_delete_choosen)
#             deleteat!(vs_t_old[2], index_delete_choosen)
#             deleteat!(v_t_old[2], index_delete_choosen)
#             deleteat!(vc_timestep[2], index_delete_choosen)
#             deleteat!(vs_timestep[2], index_delete_choosen)
#             deleteat!(v_timestep[2], index_delete_choosen)
#             deleteat!(norm_mark, index_delete_choosen)
#             deleteat!(XZ_mark_sort, index_delete_choosen)

#             # Preallocate index_delete_compo
#             index_delete_compo = Vector{Int64}(undef, nb_el * length(index_delete_choosen))

#             @inbounds for (i, k) in enumerate(index_delete_choosen)
#                 for l in 1:nb_el
#                     # Calculate the index in the preallocated array
#                     index = nb_el * (i - 1) + l
#                     index_delete_compo[index] = nb_el * (k - 1) + l
#                 end
#             end

#             deleteat!(u_mark, index_delete_compo)
#         end
#     end
# end


# norm_mark[1:20] = [0.9941639703791322, 0.9941639703791322, 0.9941639703791322, 0.9941639703791322, 0.9941639703791322, 0.9941639703791322, 0.9941639703791322, 0.9941639703791322, 0.9941639703791322, 0.9941639703791322, 0.9941639703791322, 0.9941639703791322, 0.9941639703791322, 0.9941639703791322, 0.9941639703791322, 0.9941639703791322, 0.9941639703791322, 0.9941639703791322, 0.9941639703791322, 0.9941639703791322]


# @unpack nx, nz, nx_marker, nz_marker, X_mark_save, Z_mark_save, vc_t_old, vs_t_old, v_t_old, vc_timestep, vs_timestep, v_timestep, norm_mark, XZ_mark_cell, mark_per_cell, mark_per_cell_array, mark_per_cell_array_el, u0, ratio_marker_x, ratio_marker_z_adjoint, XZ_mark_sort = advection_algo

# threshold_density = round(0.25*nx_marker*nz_marker / (nx*nz))

# i = 100
# j = 50


# index_previous = size(X_mark, 1)
# nb_el = size(u0, 3)

# # add evenly spaced markers in the cell on all arrays
# XZ_mark_cell .= ((range(zs_ad_vec[i], length=round(Int, nz_marker/nz), stop= zs_ad_vec[i+1]))' .*  ratio_marker_x)[:]
# append!(Z_mark, XZ_mark_cell)
# XZ_mark_cell .= (range(xs_ad_vec[j], length=round(Int, nx_marker/nx), stop= xs_ad_vec[j+1]) .*  ratio_marker_z_adjoint)[:]
# append!(X_mark, XZ_mark_cell)
# append!(Z_mark_save, mark_per_cell_array)
# append!(X_mark_save, mark_per_cell_array)
# append!(vc_t_old[1], mark_per_cell_array)
# append!(vs_t_old[1], mark_per_cell_array)
# append!(v_t_old[1], mark_per_cell_array)
# append!(vc_timestep[1], mark_per_cell_array)
# append!(vs_timestep[1], mark_per_cell_array)
# append!(v_timestep[1], mark_per_cell_array)
# append!(vc_t_old[2], mark_per_cell_array)
# append!(vs_t_old[2], mark_per_cell_array)
# append!(v_t_old[2], mark_per_cell_array)
# append!(vc_timestep[2], mark_per_cell_array)
# append!(vs_timestep[2], mark_per_cell_array)
# append!(v_timestep[2], mark_per_cell_array)
# append!(u_mark, mark_per_cell_array_el)

# # push each value of XZ_mark_cell in XZ_mark_sort
# append!(XZ_mark_sort, [(0.0, 0.0) for _ in axes(XZ_mark_cell, 1)])

# prev_X_mark = @view X_mark[1:index_previous]
# prev_Z_mark = @view Z_mark[1:index_previous]

# new_X_mark = @view X_mark[index_previous+1:end]
# new_Z_mark = @view Z_mark[index_previous+1:end]


# k = 1

# norm_mark .= sqrt.((prev_X_mark .- new_X_mark[k]).^2 .+ (prev_Z_mark .- new_Z_mark[k]).^2)

# norm_mark_copy = copy(norm_mark)

# @inbounds for l in axes(norm_mark, 1)
#     norm_mark[l] = sqrt(sum(abs2, ((prev_X_mark[l] - new_X_mark[k]), (prev_Z_mark[l] - new_Z_mark[k]))))
# end

# norm_mark_copy == norm_mark

# function norm_test(prev_X_mark, prev_Z_mark, new_X_mark, new_Z_mark, norm_mark)
#     k = 1
#     norm_mark .= sqrt.((prev_X_mark .- new_X_mark[k]).^2 .+ (prev_Z_mark .- new_Z_mark[k]).^2)
# end

# function norm_test2(prev_X_mark, prev_Z_mark, new_X_mark, new_Z_mark, norm_mark)
#     k = 1
#     @inbounds for l in axes(norm_mark, 1)
#         norm_mark[l] = sqrt(sum(abs2, ((prev_X_mark[l] - new_X_mark[k]), (prev_Z_mark[l] - new_Z_mark[k]))))
#     end

# end

# @btime norm_test($prev_X_mark, $prev_Z_mark, $new_X_mark, $new_Z_mark, $norm_mark)
# @btime norm_test2($prev_X_mark, $prev_Z_mark, $new_X_mark, $new_Z_mark, $norm_mark)

# index_minimum = findmin(norm_mark)[2]
# index_minimum = argmin(norm_mark)


# # rewrite the same using norm from LinearAlgebra

# @time norm_mark .= norm.(prev_X_mark .- new_X_mark[k], prev_Z_mark .- new_Z_mark[k])




# using Cthulhu

# @descend ChemicalAdvectionPorosityWave.add_marker_per_cell!(u_mark, X_mark, Z_mark, density_mark, xs_ad, zs_ad, advection_algo)

# @unpack nx, nz, nx_marker, nz_marker, X_mark_save, Z_mark_save, vc_t_old, vs_t_old, v_t_old, vc_timestep, vs_timestep, v_timestep, norm_mark, XZ_mark_cell, mark_per_cell, mark_per_cell_array, mark_per_cell_array_el, u0 = advection_algo


#     i, j = 100, 50

#         index_previous = size(X_mark, 1)
#         nb_el = size(u0, 3)

#         # add evenly spaced markers in the cell on all arrays
#         XZ_mark_cell .= ((range(zs_ad[i], length=round(Int, nz_marker/nz), stop= zs_ad[i+1]))' .*  ones(round(Int,nx_marker/nx)))[:]
#         append!(Z_mark, XZ_mark_cell)
#         XZ_mark_cell .= (range(xs_ad[j], length=round(Int, nx_marker/nx), stop= xs_ad[j+1]) .*  ones(round(Int,nz_marker/nz))')[:]
#         append!(X_mark, (range(xs_ad[j], length=round(Int, nx_marker/nx), stop= xs_ad[j+1]) .*  ones(round(Int,nz_marker/nz))')[:])
#         append!(Z_mark_save, mark_per_cell_array)
#         append!(X_mark_save, mark_per_cell_array)
#         append!(vc_t_old[1], mark_per_cell_array)
#         append!(vs_t_old[1], mark_per_cell_array)
#         append!(v_t_old[1], mark_per_cell_array)
#         append!(vc_timestep[1], mark_per_cell_array)
#         append!(vs_timestep[1], mark_per_cell_array)
#         append!(v_timestep[1], mark_per_cell_array)
#         append!(vc_t_old[2], mark_per_cell_array)
#         append!(vs_t_old[2], mark_per_cell_array)
#         append!(v_t_old[2], mark_per_cell_array)
#         append!(vc_timestep[2], mark_per_cell_array)
#         append!(vs_timestep[2], mark_per_cell_array)
#         append!(v_timestep[2], mark_per_cell_array)
#         append!(u_mark, mark_per_cell_array_el)

#         prev_X_mark = @view X_mark[1:index_previous]
#         prev_Z_mark = @view Z_mark[1:index_previous]

#         new_X_mark = @view X_mark[index_previous+1:end]
#         new_Z_mark = @view Z_mark[index_previous+1:end]

#         k = 2

#         norm_mark .= (prev_X_mark .- new_X_mark[k]).^2 .+ (prev_Z_mark .- new_Z_mark[k]).^2







#         index_delete_compo::Vector{Int64} = []
#         @inbounds for k in index_delete
#             @inbounds for l in 1:nb_el
#                 push!(index_delete_compo, nb_el * (k - 1) + l)
#             end
#         end 


# # compo_0 = copy(domain.compo_f)

# # for I in CartesianIndices(domain.compo_f[:,:,1])
# #     r = sqrt((grid.grid[1][I] - x0)^2 + (grid.grid[2][I] - z0)^2)
# #     if r <= R
# #         for k in 1:size(domain.compo_f, 3)
# #             compo_0[I[1], I[2], k] = compo_basalt[k]
# #         end
# #     else
# #         for k in 1:size(domain.compo_f, 3)
# #             compo_0[I[1], I[2], k] = compo_andesit[k]
# #         end
# #     end
# # end

# # mass_ini = sum(domain.ϕ .* compo_0[:,:,1])
# # mass_final = sum(sol[end][:,:,2] .* model.compaction_ϕ .* domain.compo_f[:,:,1])

# # println("Mass conservation: $(mass_final / mass_ini * 100) %")






# test = ones(10)
# resize!(test, st100)