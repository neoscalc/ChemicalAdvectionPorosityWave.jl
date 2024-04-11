using Parameters
using Interpolations
import StatsBase:sample

const THRESHOLD_FACTOR_DENSITY_ADD = 0.5
const THRESHOLD_FACTOR_DENSITY_REMOVE = 2.0


@with_kw struct MICScheme
    u0::Array{Float64, 3}
    nx::Int
    nz::Int
    Lx::Float64
    Lz::Float64
    Δx::Float64
    Δz::Float64
    multiplier::Int = 5
    mark_per_cell::Int = multiplier*multiplier
    mark_per_cell_array::Array{Float64, 1} = zeros(mark_per_cell)
    mark_per_cell_array_el::Array{Float64, 1} = zeros(mark_per_cell*size(u0,3))
    nx_marker::Int = (nx+1)*multiplier
    nz_marker::Int = (nz+1)*multiplier
    x_mark::Array{Float64, 1} = collect(range(0, length=nx_marker, stop= Lx))
    z_mark::Array{Float64, 1} = collect(range(0, length=nz_marker, stop= Lz))
    X_mark::Array{Float64, 1} = (x_mark' .* ones(length(z_mark)))[:]
    Z_mark::Array{Float64, 1} = (ones(length(x_mark))' .* z_mark)[:]
    X_mark_save::Array{Float64, 1} = similar(X_mark)
    Z_mark_save::Array{Float64, 1} = similar(Z_mark)
    XZ_mark_cell::Array{Float64, 1} = ones(multiplier*multiplier)
    norm_mark::Array{Float64, 1} = similar(X_mark)
    u_mark::Array{Float64, 1} = zeros(length(X_mark), size(u0,3))[:]  # temporary array
    vc_f::NamedTuple{(:x, :z), Tuple{Matrix{Float64}, Matrix{Float64}}} = (x=zeros(nz, nx),z=zeros(nz, nx))
    vc_t_old::Tuple{Vector{Float64}, Vector{Float64}} = (similar(X_mark),similar(Z_mark))
    vs_t_old::Tuple{Vector{Float64}, Vector{Float64}} = (similar(X_mark),similar(Z_mark))
    v_t_old::Tuple{Vector{Float64}, Vector{Float64}} = (similar(X_mark),similar(Z_mark))
    vc_timestep::Tuple{Vector{Float64}, Vector{Float64}} = (similar(X_mark),similar(Z_mark))
    vs_timestep::Tuple{Vector{Float64}, Vector{Float64}} = (similar(X_mark),similar(Z_mark))
    v_timestep::Tuple{Vector{Float64}, Vector{Float64}} = (similar(X_mark),similar(Z_mark))
    u_sum::Array{Float64, 2} = similar(u0[:,:,1])
    u_sum_atomic::Matrix{Atomic{Float64}} = [Atomic{Float64}(0.0) for _ in 1:size(u_sum,1), _ in 1:size(u_sum,2)]  # Assuming nx and nz are the dimensions of your 2D grid
    wt_sum::Array{Float64, 2} = similar(u0[:,:,1])
    wt_sum_atomic::Matrix{Atomic{Float64}} = [Atomic{Float64}(0.0) for _ in 1:size(wt_sum,1), _ in 1:size(wt_sum,2)]
    density_mark::Array{Float64, 2} = zeros(nz, nx)
    ratio_marker_x::Array{Float64, 1} = ones(round(Int,nx_marker/nx))
    ratio_marker_z::Array{Float64, 1} = ones(round(Int,nz_marker/nz))
    algo_name::String = "MIC"
end

function MIC_convert_adimensional(MIC, compaction_l)
    @unpack x_mark, z_mark, X_mark, Z_mark = MIC

    x_mark .= x_mark ./ compaction_l; z_mark .= z_mark ./ compaction_l
    X_mark .= X_mark ./ compaction_l; Z_mark .= Z_mark ./ compaction_l
end


function MIC_initialize_markers!(u_mark, u, MIC, parameters)

    @unpack X_mark, Z_mark, Δx, Δz, Lx, Lz, nx, nz = MIC
    @unpack x_ad, z_ad = parameters

    for k = axes(u,3)
        # interpolation
        itp = interpolate(u[:,:,k], BSpline(Linear(Periodic(OnCell()))))
        # scaled interpolation
        sitp = scale(itp, z_ad, x_ad)
        setp = extrapolate(sitp, Periodic())
        # scaled interpolation with extrapolation on the boundaries

        @inbounds @threads for I = CartesianIndices(X_mark)

            # index of the chemical element in u_mark
            nb_el = length(u_mark) ÷ length(X_mark)
            iu_mark = nb_el * (I[1] - 1) + k

            u_mark[iu_mark] = setp(Z_mark[I], X_mark[I])
        end
    end
end


function density_marker_per_cell!(density_mark, X_mark, Z_mark, xs_ad_vec, zs_ad_vec)
    # Create a list of all markers and sort it
    markers = sort(collect(zip(X_mark, Z_mark)), by = x -> (x[1], x[2]))

    @inbounds @threads for I in CartesianIndices(density_mark)
        i, j = Tuple(I)

        # Use binary search to find the range of markers within the cell's boundaries
        lower = searchsortedfirst(markers, (xs_ad_vec[j], zs_ad_vec[i]), by = x -> (x[1], x[2]))
        upper = searchsortedlast(markers, (xs_ad_vec[j+1], zs_ad_vec[i+1]), by = x -> (x[1], x[2]))

        # Count the markers within the cell
        count = 0
        for k in lower:upper
            x, z = markers[k]
            if x >= xs_ad_vec[j] && x <= xs_ad_vec[j+1] && z >= zs_ad_vec[i] && z <= zs_ad_vec[i+1]
                count += 1
            end
        end

        density_mark[I] = count
    end
end

function calculate_indices(index, nb_el)
    start_index = nb_el * (index - 1) + 1
    end_index = start_index + nb_el - 1
    return start_index, end_index
end

function add_marker_per_cell!(u_mark, X_mark, Z_mark, density_mark, xs_ad_vec, zs_ad_vec, MIC)

    @unpack nx, nz, nx_marker, nz_marker, X_mark_save, Z_mark_save, vc_t_old, vs_t_old, v_t_old, vc_timestep, vs_timestep, v_timestep, norm_mark, XZ_mark_cell, mark_per_cell, mark_per_cell_array, mark_per_cell_array_el, u0, ratio_marker_x, ratio_marker_z = MIC

    @inbounds for I in CartesianIndices(density_mark)
        i, j = Tuple(I)
        if density_mark[i, j] < round(THRESHOLD_FACTOR_DENSITY_ADD*nx_marker*nz_marker / (nx*nz))

            index_previous = size(X_mark, 1)
            nb_el = size(u0, 3)

            # add evenly spaced markers in the cell on all arrays
            XZ_mark_cell .= ((range(zs_ad_vec[i], length=round(Int, nz_marker/nz), stop= zs_ad_vec[i+1]))' .*  ratio_marker_x)[:]
            append!(Z_mark, XZ_mark_cell)
            XZ_mark_cell .= (range(xs_ad_vec[j], length=round(Int, nx_marker/nx), stop= xs_ad_vec[j+1]) .*  ratio_marker_z')[:]
            append!(X_mark, (range(xs_ad_vec[j], length=round(Int, nx_marker/nx), stop= xs_ad_vec[j+1]) .*  ones(round(Int,nz_marker/nz))')[:])
            append!(Z_mark_save, mark_per_cell_array)
            append!(X_mark_save, mark_per_cell_array)
            append!(vc_t_old[1], mark_per_cell_array)
            append!(vs_t_old[1], mark_per_cell_array)
            append!(v_t_old[1], mark_per_cell_array)
            append!(vc_timestep[1], mark_per_cell_array)
            append!(vs_timestep[1], mark_per_cell_array)
            append!(v_timestep[1], mark_per_cell_array)
            append!(vc_t_old[2], mark_per_cell_array)
            append!(vs_t_old[2], mark_per_cell_array)
            append!(v_t_old[2], mark_per_cell_array)
            append!(vc_timestep[2], mark_per_cell_array)
            append!(vs_timestep[2], mark_per_cell_array)
            append!(v_timestep[2], mark_per_cell_array)
            append!(u_mark, mark_per_cell_array_el)

            prev_X_mark = @view X_mark[1:index_previous]
            prev_Z_mark = @view Z_mark[1:index_previous]

            new_X_mark = @view X_mark[index_previous+1:end]
            new_Z_mark = @view Z_mark[index_previous+1:end]

            @inbounds for k in axes(new_X_mark, 1)

                norm_mark .= (prev_X_mark .- new_X_mark[k]).^2 .+ (prev_Z_mark .- new_Z_mark[k]).^2
                index_minimum = argmin(norm_mark)

                # replace values in u_mark with the closest marker
                index_u_mark_start, index_u_mark_end = calculate_indices(index_previous+k, nb_el)
                index_u_mark_start_near, index_u_mark_end_near = calculate_indices(index_minimum, nb_el)

                u_mark[index_u_mark_start:index_u_mark_end] .= u_mark[index_u_mark_start_near:index_u_mark_end_near]
            end

            # delete all previous markers inside this cell
            index_delete::Vector{Int64} = []

            @inbounds for k in axes(prev_X_mark, 1)
                if prev_Z_mark[k] >= zs_ad_vec[i] && prev_Z_mark[k] <= zs_ad_vec[i+1] && prev_X_mark[k] >= xs_ad_vec[j] && prev_X_mark[k] <= xs_ad_vec[j+1]
                    push!(index_delete, k)
                end
            end

            # now can append norm_mark to the previous norm_mark
            append!(norm_mark, mark_per_cell_array)

            # delete all previous markers inside this cell
            deleteat!(X_mark, index_delete)
            deleteat!(Z_mark, index_delete)
            deleteat!(Z_mark_save, index_delete)
            deleteat!(X_mark_save, index_delete)
            deleteat!(vc_t_old[1], index_delete)
            deleteat!(vs_t_old[1], index_delete)
            deleteat!(v_t_old[1], index_delete)
            deleteat!(vc_timestep[1], index_delete)
            deleteat!(vs_timestep[1], index_delete)
            deleteat!(v_timestep[1], index_delete)
            deleteat!(vc_t_old[2], index_delete)
            deleteat!(vs_t_old[2], index_delete)
            deleteat!(v_t_old[2], index_delete)
            deleteat!(vc_timestep[2], index_delete)
            deleteat!(vs_timestep[2], index_delete)
            deleteat!(v_timestep[2], index_delete)
            deleteat!(norm_mark, index_delete)

            # Preallocate index_delete_compo
            index_delete_compo = Vector{Int64}(undef, nb_el * length(index_delete))

            @inbounds for (i, k) in enumerate(index_delete)
                for l in 1:nb_el
                    # Calculate the index in the preallocated array
                    index = nb_el * (i - 1) + l
                    index_delete_compo[index] = nb_el * (k - 1) + l
                end
            end

            deleteat!(u_mark, index_delete_compo)
        end
    end
end


function remove_marker_per_cell!(u_mark, X_mark, Z_mark, density_mark, x, z, MIC)

    @unpack nx, nz, nx_marker, nz_marker, X_mark_save, Z_mark_save, vc_t_old, vs_t_old, v_t_old, vc_timestep, vs_timestep, v_timestep, norm_mark, u0 = MIC

    @inbounds for I in CartesianIndices(density_mark)
        i, j = Tuple(I)
        if density_mark[i, j] > round(THRESHOLD_FACTOR_DENSITY_REMOVE*nx_marker*nz_marker / (nx*nz))

            nb_el = size(u0, 3)

            # delete all previous markers inside this cell
            index_delete::Vector{Int64} = []

            @inbounds for k in axes(X_mark, 1)
                if Z_mark[k] >= z[i] && Z_mark[k] <= z[i+1] && X_mark[k] >= x[j] && X_mark[k] <= x[j+1]
                    push!(index_delete, k)
                end
            end

            # keep only one third of the element in index_delete but randomly

            index_delete = index_delete[1:round(Int, length(index_delete)/3)]

            # choose only one quarter of the markers in the cell
            p = 1/4
            marker_to_remove = round(Int, p*length(index_delete))
            index_delete_choosen::Vector{Int64} = sort(sample(index_delete, marker_to_remove; replace=false))

            deleteat!(X_mark, index_delete_choosen)
            deleteat!(Z_mark, index_delete_choosen)
            deleteat!(Z_mark_save, index_delete_choosen)
            deleteat!(X_mark_save, index_delete_choosen)
            deleteat!(vc_t_old[1], index_delete_choosen)
            deleteat!(vs_t_old[1], index_delete_choosen)
            deleteat!(v_t_old[1], index_delete_choosen)
            deleteat!(vc_timestep[1], index_delete_choosen)
            deleteat!(vs_timestep[1], index_delete_choosen)
            deleteat!(v_timestep[1], index_delete_choosen)
            deleteat!(vc_t_old[2], index_delete_choosen)
            deleteat!(vs_t_old[2], index_delete_choosen)
            deleteat!(v_t_old[2], index_delete_choosen)
            deleteat!(vc_timestep[2], index_delete_choosen)
            deleteat!(vs_timestep[2], index_delete_choosen)
            deleteat!(v_timestep[2], index_delete_choosen)
            deleteat!(norm_mark, index_delete_choosen)

            # Preallocate index_delete_compo
            index_delete_compo = Vector{Int64}(undef, nb_el * length(index_delete_choosen))

            @inbounds for (i, k) in enumerate(index_delete_choosen)
                for l in 1:nb_el
                    # Calculate the index in the preallocated array
                    index = nb_el * (i - 1) + l
                    index_delete_compo[index] = nb_el * (k - 1) + l
                end
            end

            deleteat!(u_mark, index_delete_compo)
        end
    end
end


function reseeding_marker!(u_mark, X_mark, Z_mark, density_mark, MIC, parameters)

    @unpack nx, nz, nx_marker, nz_marker, Lz, Lx = MIC
    @unpack xs_ad, zs_ad, xs_ad_vec, zs_ad_vec = parameters

    # density_marker_per_cell!(density_mark, X_mark, Z_mark, xs_ad_vec, zs_ad_vec)

    add_marker_per_cell!(u_mark, X_mark, Z_mark, density_mark, xs_ad_vec, zs_ad_vec, MIC)
    remove_marker_per_cell!(u_mark, X_mark, Z_mark, density_mark, xs_ad_vec, zs_ad_vec, MIC)
end


function interpol_velocity_MIC!(X_mark, Z_mark, v_staggered, v_centered, MIC, Δt, zs_ad, xs_ad, z_ad, x_ad, zs_ad_vec, xs_ad_vec)

    @unpack nx_marker, nz_marker, x_mark, z_mark, X_mark_save, Z_mark_save, vc_t_old, vs_t_old, v_t_old, vc_timestep, vs_timestep, v_timestep = MIC

    # interpolate velocities on the pressure nodes position
    setp_vcx = extrapolate(scale(interpolate(v_centered[:x], BSpline(Cubic(Periodic(OnCell())))), z_ad, x_ad), Periodic())

    # interpolation
    setp_vcz = extrapolate(scale(interpolate(v_centered[:z], BSpline(Cubic(Periodic(OnCell())))), z_ad, x_ad), Periodic())

    # interpolate velocities on the velocity nodes position
    setp_vsx = extrapolate(scale(interpolate(v_staggered[:x], BSpline(Cubic(Periodic(OnCell())))), z_ad, xs_ad), Periodic())

    # interpolation
    setp_vsz = extrapolate(scale(interpolate(v_staggered[:z], BSpline(Cubic(Periodic(OnCell())))), zs_ad, x_ad), Periodic())

    @inbounds @threads for I = CartesianIndices(X_mark)
        # interpolate velocities on the marker position from the pressure nodes
        vc_t_old[1][I] = setp_vcx(Z_mark[I], X_mark[I])
        vc_t_old[2][I] = setp_vcz(Z_mark[I], X_mark[I])

        # interpolate velocities on the velocity nodes from the velocity nodes
        vs_t_old[1][I] = setp_vsx(Z_mark[I], X_mark[I])
        vs_t_old[2][I] = setp_vsz(Z_mark[I], X_mark[I])

        # use LinP interpolation scheme (see Gerya 2019) on the marker position at the previous timestep
        v_t_old[1][I] = 2/3 * vs_t_old[1][I] + 1/3 * vc_t_old[1][I]
        v_t_old[2][I] = 2/3 * vs_t_old[2][I] + 1/3 * vc_t_old[2][I]
    end

    # save current marker position
    X_mark_save .= X_mark
    Z_mark_save .= Z_mark

    @inbounds @threads for I in eachindex(X_mark)
        X_mark[I] += v_t_old[1][I] * Δt * 0.5
        Z_mark[I] += v_t_old[2][I] * Δt * 0.5
    end

    @inbounds @threads for I = CartesianIndices(X_mark)
        vc_timestep[1][I] = setp_vcx(Z_mark[I], X_mark[I])
        vc_timestep[2][I] = setp_vcz(Z_mark[I], X_mark[I])

        vs_timestep[1][I] = setp_vsx(Z_mark[I], X_mark[I])
        vs_timestep[2][I] = setp_vsz(Z_mark[I], X_mark[I])

        # use LinP interpolation scheme (see Gerya 2019)
        v_timestep[1][I] = 2/3 * vs_timestep[1][I] + 1/3 * vc_timestep[1][I]
        v_timestep[2][I] = 2/3 * vs_timestep[2][I] + 1/3 * vc_timestep[2][I]
    end

    # use new velocity to advect marker from the previous position
    @inbounds @threads for I in eachindex(X_mark)
        X_mark[I] = X_mark_save[I] + v_timestep[1][I] * Δt
        Z_mark[I] = Z_mark_save[I] + v_timestep[2][I] * Δt
    end

    X_ad = xs_ad_vec[end] - xs_ad_vec[1]
    Z_ad = zs_ad_vec[end] - zs_ad_vec[1]

    # apply periodic boundary condition
    @inbounds for I = CartesianIndices(X_mark)
        if X_mark[I] < xs_ad[1]
            X_mark[I] += X_ad
        elseif X_mark[I] > xs_ad[end]
            X_mark[I] -= X_ad
        end
        if Z_mark[I] < zs_ad[1]
            Z_mark[I] += Z_ad
        elseif Z_mark[I] > zs_ad[end]
            Z_mark[I] -= Z_ad
        end
    end
end


function interpol_marker_to_nodes!(u, MIC, grid, parameters, z_ad_vec, x_ad_vec)

    @unpack nx, nz, Δx, Δz = grid;
    @unpack Δx_ad, Δz_ad = parameters;
    @unpack u_mark, X_mark, Z_mark, u_sum, wt_sum, u_sum_atomic, wt_sum_atomic = MIC

    for k = axes(u, 3)

        # reset values for new interpolation
        u_sum .= 0
        wt_sum .= 0

        @inbounds @threads for I in eachindex(u_sum_atomic)
            u_sum_atomic[I].value = 0
            wt_sum_atomic[I].value = 0
        end

        @inbounds @threads for I in CartesianIndices(X_mark)

            # index of markers related to the grid
            i = floor(Int, (Z_mark[I] - z_ad_vec[1]) / Δz_ad) + 1
            j = floor(Int, (X_mark[I] - x_ad_vec[1]) / Δx_ad) + 1

            # boundary conditions
            is = limit_periodic(i, nz)
            in = limit_periodic(i+1, nz)
            jw = limit_periodic(j, nx)
            je = limit_periodic(j+1, nx)

            # index of the chemical element in u_mark
            nb_el = length(u_mark) ÷ length(X_mark)
            iu_mark = nb_el * (I[1] - 1) + k

            # compute distance
            Δz_mark = Z_mark[I] - z_ad_vec[is]
            Δx_mark = X_mark[I] - x_ad_vec[jw]

            # compute weights
            wt_sw = (1 - Δx_mark / Δx_ad) * (1 - Δz_mark / Δz_ad)
            wt_se = (Δx_mark / Δx_ad) * (1 - Δz_mark / Δz_ad)
            wt_nw = (1 - Δx_mark / Δx_ad) * (Δz_mark / Δz_ad)
            wt_ne = (Δx_mark / Δx_ad) * (Δz_mark / Δz_ad)

            # Use atomic_add! to safely update u_sum and wt_sum
            atomic_add!(u_sum_atomic[is,jw], u_mark[iu_mark] * wt_sw)
            atomic_add!(u_sum_atomic[is,je], u_mark[iu_mark] * wt_se)
            atomic_add!(u_sum_atomic[in,jw], u_mark[iu_mark] * wt_nw)
            atomic_add!(u_sum_atomic[in,je], u_mark[iu_mark] * wt_ne)

            atomic_add!(wt_sum_atomic[is,jw], wt_sw)
            atomic_add!(wt_sum_atomic[is,je], wt_se)
            atomic_add!(wt_sum_atomic[in,jw], wt_nw)
            atomic_add!(wt_sum_atomic[in,je], wt_ne)
        end

        # Copy the atomic values back to the u_sum and wt_sum arrays
        @inbounds @threads for I in eachindex(u_sum)
            u_sum[I] = u_sum_atomic[I].value
            wt_sum[I] = wt_sum_atomic[I].value
        end

        @inbounds @threads for I in CartesianIndices(wt_sum)
            i, j = Tuple(I)
            if wt_sum[I] > 0
                u[i,j,k] = u_sum[I] / wt_sum[I]
            end
        end
    end
end


function MIC!(u, v_staggered, v_centered, MIC, parameters, Δt, grid)

    @unpack Δx, Δz, Lx, Lz, nx, nz = grid
    @unpack z_ad, x_ad, zs_ad, xs_ad, z_ad_vec, x_ad_vec, zs_ad_vec, xs_ad_vec = parameters
    @unpack X_mark, Z_mark = MIC

    # Interpol velocity on markers' position
    interpol_velocity_MIC!(X_mark, Z_mark, v_staggered, v_centered, MIC, Δt, zs_ad, xs_ad, z_ad, x_ad, zs_ad_vec, xs_ad_vec)

    # Perform the interpolation from the markers to the grid
    interpol_marker_to_nodes!(u, MIC, grid, parameters, z_ad_vec, x_ad_vec)
end