module SL_scheme

using Parameters
using Interpolations


@inline function limit_periodic(a, n)
    # check if index is on the boundary, if yes take value on the opposite for periodic, if not, don't change the value
    a > n ? a = a-n : a < 1 ? a = a+n : a = a
end


# second order implicit mid point method
function interpol_velocity!(v_t_half, v, v_timestep, SL, Param)

    @unpack grid, Δt, x, y, nx, ny = Param;
    @unpack x_t_half, y_t_half = SL;

    x_t_half .= grid[1] .- v[1] .* Δt/2  # calculate the previous position of the particle at position t+1/2 from the position at t+1
    y_t_half .= grid[2] .- v[2] .* Δt/2  # calculate the previous position of the particle at position t+1/2 from the position at t+1

    # for the 1st timestep, when v_timestep = 0
    if sum(v_timestep[1] .+ v_timestep[2]) !== 0.0
    # interpolation
        itp_x = interpolate(((v[1] .+ v_timestep[1]) / 2), BSpline(Linear()))
        itp_y = interpolate(((v[2] .+ v_timestep[2]) / 2), BSpline(Linear()))
    else
        itp_x = interpolate(v[1], BSpline(Linear()))
        itp_y = interpolate(v[2], BSpline(Linear()))
    end

    # scaled interpolation
    sitp = scale(itp_x, y, x)
    setp = extrapolate(sitp, Line())
    # # scaled interpolation with extrapolation on the boundaries

    for I in CartesianIndices((ny, nx))
        v_t_half[1][I] = setp(grid[2][I],x_t_half[I]);
    end

    # scaled interpolation
    sitp = scale(itp_y, y, x)
    setp = extrapolate(sitp, Line())
    # # scaled interpolation with extrapolation on the boundaries

    for I in CartesianIndices((ny, nx))
        v_t_half[2][I] = setp(y_t_half[I], grid[1][I],);
    end

    # iterate 3 times to improve approximation
    for _ in 1:3
        SL.x_t_half .= grid[1] .- v_t_half[1] .* Δt/2  # calculate the previous position of the particle at position t+1/2 from the position at t+1
        SL.y_t_half .= grid[2] .- v_t_half[2] .* Δt/2  # calculate the previous position of the particle at position t+1/2 from the position at t+1

        itp_x = interpolate(v_t_half[1], BSpline(Linear()))
        itp_y = interpolate(v_t_half[2], BSpline(Linear()))

        # scaled interpolation
        sitp = scale(itp_x, y, x)
        setp = extrapolate(sitp, Line())
        # # scaled interpolation with extrapolation on the boundaries

        @inbounds for I in CartesianIndices((ny, nx))
            v_t_half[1][I] = setp(grid[2][I],SL.x_t_half[I]);
        end

        # scaled interpolation
        sitp = scale(itp_y, y, x)
        setp = extrapolate(sitp, Line())
        # # scaled interpolation with extrapolation on the boundaries

        @inbounds for I in CartesianIndices((ny, nx))
            v_t_half[2][I] = setp(y_t_half[I], grid[1][I]);
        end
    end
end


function SL_linear!(u, SL, parameters)

    @unpack x, y, nx, ny = parameters;

    # interpolation
    itp = interpolate(u, BSpline(Linear(Periodic(OnCell()))))
    # scaled interpolation
    sitp = scale(itp, y, x)
    setp = extrapolate(sitp, Periodic())
    # scaled interpolation with extrapolation on the boundaries

    @inbounds for I = CartesianIndices((ny, nx))
        u[I] = setp(SL.y_t0[I], SL.x_t0[I])
    end
end


function SL_quasi_monotone!(u, SL, parameters)

    @unpack x, y, Δx, Δy, Δt, nx, ny, u0 = parameters;

    # interpolation
    itp = interpolate(u, BSpline(Cubic(Periodic(OnCell()))))
    # scaled interpolation
    sitp = scale(itp, y, x)
    setp = extrapolate(sitp, Periodic())
    # scaled interpolation with extrapolation on the boundaries

    @inbounds for I = CartesianIndices((ny, nx))
        SL.u_cubic[I] = setp(SL.y_t0[I], SL.x_t0[I])
    end

    @inbounds for I = CartesianIndices((ny, nx))
        i,j = Tuple(I)

        # Find nodes containing each particles and apply periodic boundaries
        local_CFLx = floor(Int, (Δt * abs(SL.v_t_half[1][I]) / Δx))
        local_CFLy = floor(Int, (Δt * abs(SL.v_t_half[2][I]) / Δy))

        if SL.v_t_half[1][I] >= 0.0
            jw, je = limit_periodic(j-1 - local_CFLx, nx), limit_periodic(j - local_CFLx, nx)
        else
            jw, je = limit_periodic(j - local_CFLx, nx), limit_periodic(j+1 - local_CFLx, nx)
        end

        if SL.v_t_half[2][I] >= 0.0
            is, in = limit_periodic(i-1 - local_CFLy, ny), limit_periodic(i - local_CFLy, ny)
        else
            is, in = limit_periodic(i - local_CFLy, ny), limit_periodic(i+1 - local_CFLy, ny)
        end

        # Apply local clipping from Bermejo and Staniforth, 1992
        SL.u_max[I] = max(u[is, j], u[in, j], u[i, je], u[i, jw])
        SL.u_min[I] = min(u[is, j], u[in, j], u[i, je], u[i, jw])

        SL.u_monot[I] = min(max(SL.u_cubic[I], SL.u_min[I]), SL.u_max[I])
    end

    u .= SL.u_monot
end


function SL_cubic!(u, SL, parameters)

    @unpack x, y, nx, ny = parameters;

    # interpolation
    itp = interpolate(u, BSpline(Cubic(Periodic(OnCell()))))
    # scaled interpolation
    sitp = scale(itp, y, x)
    setp = extrapolate(sitp, Periodic())
    # scaled interpolation with extrapolation on the boundaries

    @inbounds for I = CartesianIndices((ny, nx))
        u[I] = setp(SL.y_t0[I], SL.x_t0[I])
    end

end

function mass_conservation!(u, SL, Δx, Δy, u0, nx, ny)
    # mass balance

    lambda_lagrange = 0

    Δm = sum(u .* Δx .* Δy) - sum(u0 .* Δx .* Δy)

    if Δm != 0.0

        @inbounds for I in CartesianIndices((ny, nx))
            SL.w[I] = max(0, sign(Δm) * (SL.u_cubic[I] - SL.u_linear[I])^3)
        end

        if all(y->y==0, SL.w) == false
            lambda_lagrange = Δm / sum(SL.w .* Δx .* Δy)
        else
            lambda_lagrange = 0.0
        end

    else
        lambda_lagrange = 0.0
        SL.w .= 0
    end

    return lambda_lagrange

end


function SL_quasi_monotone_conv!(u, SL, parameters)

    @unpack x, y, Δx, Δy, Δt, nx, ny, u0 = parameters;

    # interpolation with cubic spline
    itp = interpolate(u, BSpline(Cubic(Periodic(OnCell()))))
    # scaled interpolation
    sitp = scale(itp, y, x)
    # scaled interpolation with extrapolation on the boundaries
    setp = extrapolate(sitp, Periodic())

    @inbounds for I = CartesianIndices((ny, nx))
        SL.u_cubic[I] = setp(SL.y_t0[I], SL.x_t0[I])
    end

    # interpolation with linear spline
    itp = interpolate(u, BSpline(Linear(Periodic(OnCell()))))
    # scaled interpolation
    sitp = scale(itp, y, x) 
    # scaled interpolation with extrapolation on the boundaries
    setp = extrapolate(sitp, Periodic())

    @inbounds for I = CartesianIndices((ny, nx))
        SL.u_linear[I] = setp(SL.y_t0[I], SL.x_t0[I])
    end

    @inbounds for I = CartesianIndices((ny, nx))
        i,j = Tuple(I)

        # Find nodes containing each particles and apply periodic boundaries
        local_CFLx = floor(Int, (Δt * abs(SL.v_t_half[1][I]) / Δx))
        local_CFLy = floor(Int, (Δt * abs(SL.v_t_half[2][I]) / Δy))

        if SL.v_t_half[1][I] >= 0.0
            jw, je = limit_periodic(j-1 - local_CFLx, nx), limit_periodic(j - local_CFLx, nx)
        else
            jw, je = limit_periodic(j - local_CFLx, nx), limit_periodic(j+1 - local_CFLx, nx)
        end

        if SL.v_t_half[2][I] >= 0.0
            is, in = limit_periodic(i-1 - local_CFLy, ny), limit_periodic(i - local_CFLy, ny)
        else
            is, in = limit_periodic(i - local_CFLy, ny), limit_periodic(i+1 - local_CFLy, ny)
        end

        # Apply local clipping from Bermejo and Staniforth, 1992
        SL.u_max[I] = max(u[is, j], u[in, j], u[i, jw], u[i, je])
        SL.u_min[I] = min(u[is, j], u[in, j], u[i, jw], u[i, je])

        SL.u_monot[I] = min(max(SL.u_cubic[I], SL.u_min[I]), SL.u_max[I])
    end

    # Apply global mass conservation from Bermejo and Conde, 2002
    lambda_lagrange = mass_conservation!(SL.u_monot, SL, Δx, Δy, u0, nx, ny)

    u .= SL.u_monot .- lambda_lagrange .* SL.w
end


function semi_lagrangian!(u, SL, v, v_timestep, parameters; method::String="quasi-monotone")

    @unpack x, y, grid, Δx, Δy, Δt, nx, ny = parameters;

    interpol_velocity!(SL.v_t_half, v, v_timestep, SL, parameters)

    SL.x_t0 .= grid[1] .- SL.v_t_half[1] .* Δt  # calculate the previous position of the particle at position t from the position at t+1
    SL.y_t0 .= grid[2] .- SL.v_t_half[2] .* Δt  # calculate the previous position of the particle at position t from the position at t+1


    if method == "quasi-monotone conservative"
        SL_quasi_monotone_conv!(u, SL, parameters)
    elseif method == "quasi-monotone"
        SL_quasi_monotone!(u, SL, parameters)
    elseif method == "cubic"
        SL_cubic!(u, SL, parameters)
    elseif method == "linear"
        SL_linear!(u, SL, parameters)
    else
        error("Unknown method for the Semi-Lagrangian Scheme.")
    end

end


end