module MIC_scheme

using Parameters
using Interpolations
using Base.Threads: @threads, Atomic, atomic_add!

@inline function limit_periodic(a, n)
    # check if index is on the boundary, if yes take value on the opposite for periodic, if not, don't change the value
    a > n ? a = a-n : a < 1 ? a = a+n : a = a
end


function MIC_initialize_markers!(u_mark, u, MIC, parameters)

    @unpack x,y,Δt = parameters;
    @unpack X_mark, Y_mark = MIC

    # interpolation
    itp = interpolate(u, BSpline(Linear(Periodic(OnCell()))))
    # scaled interpolation
    sitp = scale(itp, y, x)
    setp = extrapolate(sitp, Periodic())
    # scaled interpolation with extrapolation on the boundaries

    @inbounds @threads for I = CartesianIndices(u_mark)
        u_mark[I] = setp(Y_mark[I], X_mark[I])
    end

end

function velocity_interp!(X_mark, Y_mark, v, MIC, parameters)

    @unpack x, y, Δt = parameters;
    @unpack X, Y, nx_marker, ny_marker, x_mark, y_mark, u_mark, X_mark_save, Y_mark_save, v_t_old, v_timestep = MIC

    # interpolation
    itp = interpolate(v[1], BSpline(Linear(Periodic(OnCell()))))
    # scaled interpolation
    sitp = scale(itp, y, x)
    setp_vx = extrapolate(sitp, Periodic())
    # scaled interpolation with extrapolation on the boundaries

    # interpolation
    itp = interpolate(v[2], BSpline(Linear(Periodic(OnCell()))))
    # scaled interpolation
    sitp = scale(itp, y, x)
    setp_vy = extrapolate(sitp, Periodic())
    # scaled interpolation with extrapolation on the boundaries

    @inbounds @threads for I = CartesianIndices(u_mark)
        v_t_old[1][I] = setp_vx(Y_mark[I], X_mark[I])
        v_t_old[2][I] = setp_vy(Y_mark[I], X_mark[I])
    end

    X_mark_save .= X_mark
    Y_mark_save .= Y_mark

    # save current marker position
    @inbounds @threads for I in eachindex(X_mark)
        X_mark[I] += v_t_old[1][I] * Δt * 0.5
        Y_mark[I] += v_t_old[2][I] * Δt * 0.5
    end

    @inbounds @threads for I = CartesianIndices(u_mark)
        v_timestep[1][I] = setp_vx(Y_mark[I], X_mark[I])
        v_timestep[2][I] = setp_vy(Y_mark[I], X_mark[I])
    end

    @inbounds @threads for I in eachindex(X_mark)
        X_mark[I] = X_mark_save[I] + v_timestep[1][I] * Δt
        Y_mark[I] = Y_mark_save[I] + v_timestep[2][I] * Δt
    end
end


function interpol_marker_to_nodes!(u, MIC, parameters)

    @unpack x, y, Δx, Δy, nx, ny, Δt = parameters;
    @unpack u_mark, X_mark, Y_mark, u_sum, wt_sum, x_vec, y_vec, u_sum_atomic, wt_sum_atomic = MIC

    # reset values of atomic arrays
    @inbounds @threads for I in eachindex(u_sum_atomic)
        u_sum_atomic[I].value = 0
        wt_sum_atomic[I].value = 0
    end

    @inbounds @threads for I in CartesianIndices(X_mark)

        # index of markers related to the grid
        i = floor(Int,(Y_mark[I])/Δy) + 1
        j = floor(Int,(X_mark[I])/Δx) + 1

        # boundary conditions
        is = limit_periodic(i, ny)
        in = limit_periodic(i+1, ny)
        jw = limit_periodic(j, nx)
        je = limit_periodic(j+1, nx)

        # compute distance
        Δy_mark = Y_mark[I] - y_vec[is]
        Δx_mark = X_mark[I] - x_vec[jw]

        # compute weights
        wt_sw = (1 - Δx_mark / Δx) * (1 - Δy_mark / Δy)
        wt_se = (Δx_mark / Δx) * (1 - Δy_mark / Δy)
        wt_nw = (1 - Δx_mark / Δx) * (Δy_mark / Δy)
        wt_ne = (Δx_mark / Δx) * (Δy_mark / Δy)

        # Use atomic_add! to safely update u_sum and wt_sum
        atomic_add!(u_sum_atomic[is,jw], u_mark[I] * wt_sw)
        atomic_add!(u_sum_atomic[is,je], u_mark[I] * wt_se)
        atomic_add!(u_sum_atomic[in,jw], u_mark[I] * wt_nw)
        atomic_add!(u_sum_atomic[in,je], u_mark[I] * wt_ne)

        atomic_add!(wt_sum_atomic[is,jw], wt_sw)
        atomic_add!(wt_sum_atomic[is,je], wt_se)
        atomic_add!(wt_sum_atomic[in,jw], wt_nw)
        atomic_add!(wt_sum_atomic[in,je], wt_ne)
    end

    # Combine the atomic arrays into the final arrays
    @inbounds @threads for I in eachindex(u_sum)
        u_sum[I] = u_sum_atomic[I].value
        wt_sum[I] = wt_sum_atomic[I].value
    end

    @inbounds @threads for I in CartesianIndices(size(u))
        if wt_sum[I] > 0
            u[I] = u_sum[I] / wt_sum[I]
        end
    end

end


function MIC!(u, MIC, v, parameters)

    @unpack X_mark, Y_mark = MIC

    velocity_interp!(X_mark, Y_mark, v, MIC, parameters)

    # Perform the interpolation from the markers to the grid
    interpol_marker_to_nodes!(u, MIC, parameters)
end

end