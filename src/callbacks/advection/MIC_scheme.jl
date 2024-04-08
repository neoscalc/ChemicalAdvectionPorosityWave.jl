using Parameters
using Interpolations
import StatsBase:sample

@with_kw struct MICScheme
    u0::Array{Float64, 3}
    nx::Int
    nz::Int
    Lx::Float64
    Lz::Float64
    Δx::Float64
    Δz::Float64
    nx_marker::Int = (nx+1)*5
    nz_marker::Int = (nz+1)*5
    x_mark::Array{Float64, 1} = collect(range(0, length=nx_marker, stop= Lx))
    z_mark::Array{Float64, 1} = collect(range(0, length=nz_marker, stop= Lz))
    X_mark::Array{Float64, 1} = (x_mark' .* ones(length(z_mark)))[:]
    Z_mark::Array{Float64, 1} = (ones(length(x_mark))' .* z_mark)[:]
    X_mark_save::Array{Float64, 1} = similar(X_mark)
    Z_mark_save::Array{Float64, 1} = similar(Z_mark)
    norm_mark::Array{Float64, 1} = similar(X_mark)
    u_mark::Array{Float64, 1} = zeros(length(X_mark), size(u0,3))[:]  # temporary array
    vc_f::NamedTuple{(:x, :z), Tuple{Matrix{Float64}, Matrix{Float64}}} = (x=zeros(nz, nx),z=zeros(nz, nx))
    vc_t_old::Tuple{Vector{Float64}, Vector{Float64}} = (similar(X_mark),similar(Z_mark))
    vs_t_old::Tuple{Vector{Float64}, Vector{Float64}} = (similar(X_mark),similar(Z_mark))
    v_t_old::Tuple{Vector{Float64}, Vector{Float64}} = (similar(X_mark),similar(Z_mark))
    vc_timestep::Tuple{Vector{Float64}, Vector{Float64}} = (similar(X_mark),similar(Z_mark))
    vs_timestep::Tuple{Vector{Float64}, Vector{Float64}} = (similar(X_mark),similar(Z_mark))
    v_timestep::Tuple{Vector{Float64}, Vector{Float64}} = (similar(X_mark),similar(Z_mark))
    u_sum::Array{Float64, 3} = similar(u0)
    wt_sum::Array{Float64, 3} = similar(u0)
    density_mark::Array{Float64, 2} = zeros(nz, nx)
    algo_name::String = "MIC"
end

function MIC_convert_adimensional(MIC, compaction_l)
    @unpack x_mark, z_mark, X_mark, Z_mark = MIC

    x_mark .= x_mark ./ compaction_l; z_mark .= z_mark ./ compaction_l
    X_mark .= X_mark ./ compaction_l; Z_mark .= Z_mark ./ compaction_l
end


function MIC_initialize_markers!(u_mark, u, MIC, compaction_l)

    @unpack X_mark, Z_mark, Δx, Δz, Lx, Lz, nx, nz = MIC

    zc_ad = range((Δz/2) / compaction_l, length=nz, stop= (Lz-Δz/2) / compaction_l)
    xc_ad = range((Δx/2) / compaction_l, length=nx, stop= (Lx-Δx/2) / compaction_l)

    for k = axes(u,3)
        # interpolation
        itp = interpolate(u[:,:,k], BSpline(Linear(Periodic(OnCell()))))
        # scaled interpolation
        sitp = scale(itp, zc_ad, xc_ad)
        setp = extrapolate(sitp, Periodic())
        # scaled interpolation with extrapolation on the boundaries

        @inbounds for I = CartesianIndices(X_mark)

            # index of the chemical element in u_mark
            nb_el = length(u_mark) ÷ length(X_mark)
            iu_mark = nb_el * (I[1] - 1) + k

            u_mark[iu_mark] = setp(Z_mark[I], X_mark[I])
        end
    end
end

function density_marker_per_cell!(density_mark, X_mark, Z_mark, x, z)

    # reset density_cell
    density_mark .= 0

    for I in CartesianIndices(density_mark)
        i, j = Tuple(I)
        for k in axes(X_mark,1)
            if X_mark[k] >= x[j] && X_mark[k] <= x[j+1] && Z_mark[k] >= z[i] && Z_mark[k] <= z[i+1]
                density_mark[i, j] += 1
            end
        end
    end
end

function add_marker_per_cell!(u_mark, X_mark, Z_mark, density_mark, x, z, MIC)

    @unpack nx, nz, nx_marker, nz_marker, X_mark_save, Z_mark_save, vc_t_old, vs_t_old, v_t_old, vc_timestep, vs_timestep, v_timestep, norm_mark = MIC

    @inbounds for I in CartesianIndices(density_mark)
        i, j = Tuple(I)
        if density_mark[i, j] < round(0.25*nx_marker*nz_marker / (nx*nz))

            index_previous = size(X_mark, 1)
            nb_el = length(u_mark) ÷ length(X_mark)
            mark_per_cell = round(Int, nx_marker/nx)*round(Int, nz_marker/nz)


            # add evenly spaced markers in the cell on all arrays
            append!(Z_mark, ((range(z[i], length=round(Int, nz_marker/nz), stop= z[i+1]))' .*  ones(round(Int,nx_marker/nx)))[:])
            append!(X_mark, (range(x[j], length=round(Int, nx_marker/nx), stop= x[j+1]) .*  ones(round(Int,nz_marker/nz))')[:])
            append!(Z_mark_save, zeros(mark_per_cell))
            append!(X_mark_save, zeros(mark_per_cell))
            append!(vc_t_old[1], zeros(mark_per_cell))
            append!(vs_t_old[1], zeros(mark_per_cell))
            append!(v_t_old[1], zeros(mark_per_cell))
            append!(vc_timestep[1], zeros(mark_per_cell))
            append!(vs_timestep[1], zeros(mark_per_cell))
            append!(v_timestep[1], zeros(mark_per_cell))
            append!(vc_t_old[2], zeros(mark_per_cell))
            append!(vs_t_old[2], zeros(mark_per_cell))
            append!(v_t_old[2], zeros(mark_per_cell))
            append!(vc_timestep[2], zeros(mark_per_cell))
            append!(vs_timestep[2], zeros(mark_per_cell))
            append!(v_timestep[2], zeros(mark_per_cell))
            append!(norm_mark, zeros(mark_per_cell))
            append!(u_mark, zeros(mark_per_cell * (nb_el)))

            prev_X_mark = @view X_mark[1:index_previous]
            prev_Z_mark = @view Z_mark[1:index_previous]

            new_X_mark = @view X_mark[index_previous+1:end]
            new_Z_mark = @view Z_mark[index_previous+1:end]

            @inbounds for k in axes(new_X_mark, 1)
                norm_mark[1:index_previous] .= (prev_X_mark .- new_X_mark[k]).^2 .+ (prev_Z_mark .- new_Z_mark[k]).^2
                index_minimum = argmin(norm_mark)

                # replace values in u_mark with the closest marker
                index_u_mark_start = nb_el * ((index_previous+k) - 1) + 1
                index_u_mark_end = nb_el * ((index_previous+k) - 1) + nb_el
                index_u_mark_start_near = nb_el * (index_minimum - 1) + 1
                index_u_mark_end_near = nb_el * (index_minimum - 1) + nb_el

                u_mark[index_u_mark_start:index_u_mark_end] .= u_mark[index_u_mark_start_near:index_u_mark_end_near]
            end

            # delete all previous markers inside this cell
            index_delete::Vector{Int64} = []

            @inbounds for k in axes(prev_X_mark, 1)
                if prev_Z_mark[k] >= z[i] && prev_Z_mark[k] <= z[i+1] && prev_X_mark[k] >= x[j] && prev_X_mark[k] <= x[j+1]
                    push!(index_delete, k)
                end
            end

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

            # index of the chemical element in u_mark
            index_delete_compo::Vector{Int64} = []
            @inbounds for k in index_delete
                @inbounds for l in 1:nb_el
                    push!(index_delete_compo, nb_el * (k - 1) + l)
                end
            end

            deleteat!(u_mark, index_delete_compo)
        end
    end
end


function remove_marker_per_cell!(u_mark, X_mark, Z_mark, density_mark, x, z, MIC)

    @unpack nx, nz, nx_marker, nz_marker, X_mark_save, Z_mark_save, vc_t_old, vs_t_old, v_t_old, vc_timestep, vs_timestep, v_timestep, norm_mark = MIC

    @inbounds for I in CartesianIndices(density_mark)
        i, j = Tuple(I)
        if density_mark[i, j] > round(2.0*nx_marker*nz_marker / (nx*nz))

            index_previous = size(X_mark, 1)
            nb_el = length(u_mark) ÷ length(X_mark)
            mark_per_cell = round(Int, nx_marker/nx)*round(Int, nz_marker/nz)

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

            # index of the chemical element in u_mark
            index_delete_compo::Vector{Int64} = []
            @inbounds for k in index_delete_choosen
                @inbounds for l in 1:nb_el
                    push!(index_delete_compo, nb_el * (k - 1) + l)
                end
            end

            deleteat!(u_mark, index_delete_compo)
        end
    end
end



function reseeding_marker!(u_mark, X_mark, Z_mark, density_mark, MIC, compaction_l)

    @unpack nx, nz, nx_marker, nz_marker, Lz, Lx = MIC

    z_ad = range(0, length=nz+1, stop= Lz / compaction_l)
    x_ad = range(0, length=nx+1, stop= Lx / compaction_l)

    density_marker_per_cell!(density_mark, X_mark, Z_mark, x_ad, z_ad)

    # @show maximum(density_mark)
    # @show minimum(density_mark)

    add_marker_per_cell!(u_mark, X_mark, Z_mark, density_mark, x_ad, z_ad, MIC)
    remove_marker_per_cell!(u_mark, X_mark, Z_mark, density_mark, x_ad, z_ad, MIC)
end



function interpol_velocity_MIC!(X_mark, Z_mark, v_staggered, v_centered, MIC, Δt, zs_ad, xs_ad, z_ad, x_ad)

    @unpack nx_marker, nz_marker, x_mark, z_mark, X_mark_save, Z_mark_save, vc_t_old, vs_t_old, v_t_old, vc_timestep, vs_timestep, v_timestep = MIC

    # interpolate velocities on the pressure nodes position
    setp_vcx = extrapolate(scale(interpolate(v_centered[:x], BSpline(Cubic(Periodic(OnCell())))), z_ad, x_ad), Periodic())

    # interpolation
    setp_vcz = extrapolate(scale(interpolate(v_centered[:z], BSpline(Cubic(Periodic(OnCell())))), z_ad, x_ad), Periodic())

    for I = CartesianIndices(X_mark)
        vc_t_old[1][I] = setp_vcx(Z_mark[I], X_mark[I])
        vc_t_old[2][I] = setp_vcz(Z_mark[I], X_mark[I])
    end

    # interpolate velocities on the velocity nodes position
    setp_vsx = extrapolate(scale(interpolate(v_staggered[:x], BSpline(Cubic(Periodic(OnCell())))), z_ad, xs_ad), Periodic())

    # interpolation
    setp_vsz = extrapolate(scale(interpolate(v_staggered[:z], BSpline(Cubic(Periodic(OnCell())))), zs_ad, x_ad), Periodic())

    @inbounds for I = CartesianIndices(X_mark)
        vs_t_old[1][I] = setp_vsx(Z_mark[I], X_mark[I])
        vs_t_old[2][I] = setp_vsz(Z_mark[I], X_mark[I])

        # use LinP interpolation scheme (see Gerya 2019)
        v_t_old[1][I] = 2/3 * vs_t_old[1][I] + 1/3 * vc_t_old[1][I]
        v_t_old[2][I] = 2/3 * vs_t_old[2][I] + 1/3 * vc_t_old[2][I]
    end

    # save current marker position
    X_mark_save .= X_mark
    Z_mark_save .= Z_mark

    # Advect marker with previous velocity
    X_mark .= X_mark .+ v_t_old[1] .* Δt
    Z_mark .= Z_mark .+ v_t_old[2] .* Δt

    @inbounds for I = CartesianIndices(X_mark)
        vc_timestep[1][I] = setp_vcx(Z_mark[I], X_mark[I])
        vc_timestep[2][I] = setp_vcz(Z_mark[I], X_mark[I])

        vs_timestep[1][I] = setp_vsx(Z_mark[I], X_mark[I])
        vs_timestep[2][I] = setp_vsz(Z_mark[I], X_mark[I])

        # use LinP interpolation scheme (see Gerya 2019)
        v_timestep[1][I] = 2/3 * vs_timestep[1][I] + 1/3 * vc_timestep[1][I]
        v_timestep[2][I] = 2/3 * vs_timestep[2][I] + 1/3 * vc_timestep[2][I]
    end

    # Advect marker with fixed point velocity
    if !iszero(v_t_old[1])
        X_mark .= X_mark_save .+ (v_t_old[1] .+ v_timestep[1]) .* 0.5 .* Δt
    else
        X_mark .= X_mark_save .+ v_timestep[1] .* Δt
    end

    if !iszero(v_t_old[2])
        Z_mark .= Z_mark_save .+ (v_t_old[2] .+ v_timestep[2]) .* 0.5 .* Δt
    else
        Z_mark .= Z_mark_save .+ v_timestep[2] .* Δt
    end

    X_ad = xs_ad[end] - xs_ad[1]
    Z_ad = zs_ad[end] - zs_ad[1]

    # apply periodic boundary condition
    for I = CartesianIndices(X_mark)
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
    @unpack u_mark, X_mark, Z_mark, u_sum, wt_sum = MIC

    for k = axes(u, 3)

        # reset values for new interpolation
        u_sum .= 0
        wt_sum .= 0

        #! Not thread safe
        @inbounds for I in CartesianIndices(X_mark)

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

            # compute sum
            u_sum[is,jw] += u_mark[iu_mark] * wt_sw
            u_sum[is,je] += u_mark[iu_mark] * wt_se
            u_sum[in,jw] += u_mark[iu_mark] * wt_nw
            u_sum[in,je] += u_mark[iu_mark] * wt_ne

            wt_sum[is,jw] += wt_sw
            wt_sum[is,je] += wt_se
            wt_sum[in,jw] += wt_nw
            wt_sum[in,je] += wt_ne
        end

        @inbounds for I in CartesianIndices(wt_sum)
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
    interpol_velocity_MIC!(X_mark, Z_mark, v_staggered, v_centered, MIC, Δt, zs_ad, xs_ad, z_ad, x_ad)

    # Perform the interpolation from the markers to the grid
    interpol_marker_to_nodes!(u, MIC, grid, parameters, z_ad_vec, x_ad_vec)

end