module MIC_scheme

using Parameters
using Interpolations

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

    @inbounds for I = CartesianIndices(u_mark)
        u_mark[I] = setp(Y_mark[I], X_mark[I])
    end

end

function velocity_interp!(X_mark, Y_mark, v, MIC, parameters)

    @unpack x, y, Δt = parameters;
    @unpack X, Y, nx_marker, ny_marker, x_mark, y_mark, X_mark, Y_mark, u_mark, X_mark_save, Y_mark_save, v_t_old, v_timestep = MIC

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

    @inbounds for I = CartesianIndices(u_mark)
        v_t_old[1][I] = setp_vx(Y_mark[I], X_mark[I])
        v_t_old[2][I] = setp_vy(Y_mark[I], X_mark[I])
    end

    # save current marker position
    X_mark_save .= X_mark
    Y_mark_save .= Y_mark

    # Advect marker with previous velocity
    X_mark .= X_mark .+ v_t_old[1] .* Δt
    Y_mark .= Y_mark .+ v_t_old[2] .* Δt

    @inbounds for I = CartesianIndices(u_mark)
        v_timestep[1][I] = setp_vx(Y_mark[I], X_mark[I])
        v_timestep[2][I] = setp_vy(Y_mark[I], X_mark[I])
    end

    # Advect marker with fixed point velocity
    X_mark .= X_mark_save .+ (v_t_old[1] .+ v_timestep[1]) .* 0.5 .* Δt
    Y_mark .= Y_mark_save .+ (v_t_old[2] .+ v_timestep[2]) .* 0.5 .* Δt

end


function interpol_marker_to_nodes!(u, MIC, parameters)

    @unpack x, y, Δx, Δy, nx, ny, Δt = parameters;
    @unpack u_mark, X_mark, Y_mark, u_sum, wt_sum = MIC

    # reset values for new interpolation
    u_sum .= 0
    wt_sum .= 0

    @inbounds for I in CartesianIndices(X_mark)

        # index of markers related to the grid
        i = trunc(Int,(Y_mark[I])/Δy) + 1
        j = trunc(Int,(X_mark[I])/Δx) + 1

        # boundary conditions
        is = limit_periodic(i, ny)
        in = limit_periodic(i+1, ny)
        jw = limit_periodic(j, nx)
        je = limit_periodic(j+1, nx)


        # compute distance
        Δy_mark = Y_mark[I] - y[is]
        Δx_mark = X_mark[I] - x[jw]

        # compute weights
        wt_sw = (1 - Δx_mark / Δx) * (1 - Δy_mark / Δy)
        wt_se = (Δx_mark / Δx) * (1 - Δy_mark / Δy)
        wt_nw = (1 - Δx_mark / Δx) * (Δy_mark / Δy)
        wt_ne = (Δx_mark / Δx) * (Δy_mark / Δy)

        # compute sum
        u_sum[is,jw] += u_mark[I] * wt_sw
        u_sum[is,je] += u_mark[I] * wt_se
        u_sum[in,jw] += u_mark[I] * wt_nw
        u_sum[in,je] += u_mark[I] * wt_ne

        wt_sum[is,jw] += wt_sw
        wt_sum[is,je] += wt_se
        wt_sum[in,jw] += wt_nw
        wt_sum[in,je] += wt_ne
    end

    @inbounds for I in CartesianIndices(size(u))
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