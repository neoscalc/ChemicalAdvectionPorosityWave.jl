module SL_scheme

using Parameters
using Interpolations
using OrdinaryDiffEq
using Base.Threads: @threads

@inline function limit_periodic(a, n)
    # check if index is on the boundary, if yes take value on the opposite for periodic, if not, don't change the value
    a > n ? a = a-n : a < 1 ? a = a+n : a = a
end


# second order implicit mid point method
function initial_pos_marker!(x_t_depart, y_t_depart, v_t_half, v, v_timestep, SL, Param, iter)

    @unpack grid, Δt, x, y, nx, ny = Param;

    if any(v_timestep[1] .!= 0.0) || any(v_timestep[2] .!= 0.0)
        v_t_half[1] .= (v[1] .+ v_timestep[1]) ./ 2
        v_t_half[2] .= (v[2] .+ v_timestep[2]) ./ 2
    else
        v_t_half[1] .= v[1]
        v_t_half[2] .= v[2]
    end

    x_t_depart .= grid[1] .- v_t_half[1] .* Δt / 2 # calculate the previous position of the particle at position t+1/2 from the position at t+1
    y_t_depart .= grid[2] .- v_t_half[2] .* Δt / 2
    itp_x = interpolate(v_t_half[1], BSpline(Linear()))
    itp_y = interpolate(v_t_half[2], BSpline(Linear()))

    # scaled interpolation
    sitp = scale(itp_x, y, x)
    setp_x = extrapolate(sitp, Line())
    # # scaled interpolation with extrapolation on the boundaries

    # scaled interpolation
    sitp = scale(itp_y, y, x)
    setp_y = extrapolate(sitp, Line())
    # # scaled interpolation with extrapolation on the boundaries

    # iterate iter times to improve approximation
    for _ in 1:iter
        @inbounds @threads for I in CartesianIndices((ny, nx))
            # calculate the previous position of the particle at position t from the position at t+1/2
            x_t_depart[I] = grid[1][I] - setp_x(grid[2][I],((x_t_depart[I] + grid[1][I]) / 2)) * Δt
            y_t_depart[I] = grid[2][I] - setp_y((y_t_depart[I] + grid[2][I]) / 2 , grid[1][I]) * Δt
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

    @inbounds @threads for I = CartesianIndices((ny, nx))
        u[I] = setp(SL.y_t_depart[I], SL.x_t_depart[I])
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

    @inbounds @threads for I = CartesianIndices((ny, nx))
        SL.u_cubic[I] = setp(SL.y_t_depart[I], SL.x_t_depart[I])
    end

    @inbounds @threads for I = CartesianIndices((ny, nx))
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

    @inbounds @threads for I = CartesianIndices((ny, nx))
        u[I] = setp(SL.y_t_depart[I], SL.x_t_depart[I])
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
        SL.u_cubic[I] = setp(SL.y_t_depart[I], SL.x_t_depart[I])
    end

    # interpolation with linear spline
    itp = interpolate(u, BSpline(Linear(Periodic(OnCell()))))
    # scaled interpolation
    sitp = scale(itp, y, x) 
    # scaled interpolation with extrapolation on the boundaries
    setp = extrapolate(sitp, Periodic())

    @inbounds for I = CartesianIndices((ny, nx))
        SL.u_linear[I] = setp(SL.y_t_depart[I], SL.x_t_depart[I])
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

    initial_pos_marker!(SL.x_t_depart, SL.y_t_depart, SL.v_t_half, v, v_timestep, SL, parameters, 6)

    if method == "quasi-monotone conservative"
        SL_quasi_monotone_conv!(u, SL, parameters)
    elseif method == "quasi-monotone"
        SL_quasi_monotone!(u, SL, parameters)
    elseif method == "cubic"
        SL_cubic!(u, SL, parameters)
    elseif method == "linear"
        SL_linear!(u, SL, parameters)
    else
        error("Unknown method for the semi-Lagrangian scheme.")
    end
end


end