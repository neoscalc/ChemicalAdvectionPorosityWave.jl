using Parameters
using Interpolations

@with_kw struct SemiLagrangianScheme
    nx::Int
    nz::Int
    u0::Array{Float64, 3}
    u_monot::Array{Float64, 3} = zeros(nz, nx, size(u0,3))
    u_cubic::Array{Float64, 3} = zeros(nz, nx, size(u0,3))
    u_linear::Array{Float64, 3} = zeros(nz, nx, size(u0,3))
    u_max::Array{Float64, 2} = zeros(nz, nx)
    u_min::Array{Float64, 2} = zeros(nz, nx)
    x_t_depart::Array{Float64, 2} = zeros(nz, nx)
    z_t_depart::Array{Float64, 2} = zeros(nz, nx)
    w::Array{Float64, 2} = zeros(nz, nx)
    v_t_half::NamedTuple{(:x, :z), Tuple{Matrix{Float64}, Matrix{Float64}}} = (x=zeros(nz, nx),z=zeros(nz, nx))
    vc_f::NamedTuple{(:x, :z), Tuple{Matrix{Float64}, Matrix{Float64}}} = (x=zeros(nz, nx),z=zeros(nz, nx))
    vc_previous::NamedTuple{(:x, :z), Tuple{Matrix{Float64}, Matrix{Float64}}} = (x=zeros(nz, nx),z=zeros(nz, nx))
    div_v::Array{Float64, 2} = zeros(nz, nx)
    det::Array{Float64, 2} = zeros(nz, nx)
    algo_name::String = "Semi-Lagrangian"
end

# second order implicit mid point method
function initial_pos_marker!(x_t_depart, z_t_depart, v_t_half, v, v_timestep, Δt, Grid, parameters)

    @unpack nx, nz = Grid;
    @unpack grid_ad, x_ad, z_ad = parameters;

    # for the 1st timestep, when v_timestep = 0
    if any(v_timestep[:x] .!= 0.0) || any(v_timestep[:z] .!= 0.0)
        @inbounds @threads for I in eachindex(v_t_half[1])
            v_t_half[:x][I] = (v[:x][I] + v_timestep[:x][I]) * 0.5
            v_t_half[:z][I] = (v[:z][I] + v_timestep[:z][I]) * 0.5
        end
    else
        v_t_half[:x] .= v[:x]
        v_t_half[:z] .= v[:z]
    end

    # calculate the previous position of the particle at position t+1/2 from the position at t+1 as first guess
    @inbounds @threads for I in eachindex(x_t_depart)
        x_t_depart[I] = grid_ad[:x][I] - v_t_half[:x][I] * Δt * 0.5
        z_t_depart[I] = grid_ad[:z][I] - v_t_half[:z][I] * Δt * 0.5
    end

    setp_x = extrapolate(scale(interpolate(v_t_half[:x], BSpline(Linear())), z_ad, x_ad), Line())
    setp_z = extrapolate(scale(interpolate(v_t_half[:z], BSpline(Linear())), z_ad, x_ad), Line())

    # iterate 4 times to improve approximation
    for _ in 1:4
        @inbounds @threads for I in eachindex(x_t_depart)
            x_t_depart[I] = grid_ad[:x][I] - setp_x(grid_ad[:z][I], (x_t_depart[I] + grid_ad[:x][I]) * 0.5) * Δt
            z_t_depart[I] = grid_ad[:z][I] - setp_z((z_t_depart[I] + grid_ad[:z][I]) * 0.5, grid_ad[:x][I]) * Δt
        end
    end
end


function SL_linear!(u_monot, SL, Grid, parameters)

    @unpack nx, nz = Grid;
    @unpack x_ad, z_ad = parameters;


    for k = axes(u_monot, 3)

        # interpolation
        itp = interpolate(u_monot[:,:,k], BSpline(Linear(Periodic(OnCell()))))
        # scaled interpolation
        sitp = scale(itp, z_ad, x_ad)
        # scaled interpolation with extrapolation on the boundaries
        setp = extrapolate(sitp, Periodic())

        for I = CartesianIndices(SL.x_t0)
            i,j = Tuple(I)
            u_monot[i,j,k] = setp(SL.z_t0[i,j], SL.x_t0[i,j])
        end

    end

end


function SL_cubic!(u_monot, SL, Grid, parameters)

    @unpack nx, nz = Grid;
    @unpack x_ad, z_ad = parameters;


    for k = axes(u_monot, 3)

        # interpolation
        itp = interpolate(u_monot[:,:,k], BSpline(Cubic(Periodic(OnCell()))))
        # scaled interpolation
        sitp = scale(itp, z_ad, x_ad)
        # scaled interpolation with extrapolation on the boundaries
        setp = extrapolate(sitp, Periodic())

        for I = CartesianIndices(SL.x_t0)
            i,j = Tuple(I)
            u_monot[i,j,k] = setp(SL.z_t0[i,j], SL.x_t0[i,j])
        end

    end

end


function SL_quasi_monotone!(u_monot, SL, Δt, Grid, parameters)

    @unpack nx, nz = Grid;
    @unpack x_ad, z_ad, Δx_ad, Δz_ad = parameters;
    @unpack u0 = SL;

    # iteration over the chemical elements
    for k = axes(u_monot, 3)

        # # cubic interpolation
        # itp = interpolate(u_monot[:,:,k], BSpline(Cubic(Periodic(OnCell()))))
        # # scaled interpolation
        # sitp = scale(itp, z_ad, x_ad)
        # # scaled interpolation with extrapolation on the boundaries
        # setp = extrapolate(sitp, Periodic())
        setp = extrapolate(scale(interpolate(u_monot[:,:,k], BSpline(Cubic(Periodic(OnCell())))), z_ad, x_ad), Periodic())

        @inbounds @threads for I = CartesianIndices(SL.x_t_depart)
            i,j = Tuple(I)
            SL.u_cubic[i,j,k] = setp(SL.z_t_depart[i,j], SL.x_t_depart[i,j])
        end

        #quasi-monoticity
        @inbounds @threads for I = CartesianIndices((nz, nx))
            i,j = Tuple(I)

            local_CFLx = floor(Int, (Δt * abs(SL.v_t_half[:x][I]) / Δx_ad))
            local_CFLz = floor(Int, (Δt * abs(SL.v_t_half[:z][I]) / Δz_ad))

            if SL.v_t_half[:x][I] >= 0.0
                jw, je = limit_periodic(j-1 - local_CFLx, nx), limit_periodic(j - local_CFLx, nx)
            else
                jw, je = limit_periodic(j - local_CFLx, nx), limit_periodic(j+1 - local_CFLx, nx)
            end

            if SL.v_t_half[:z][I] >= 0.0
                is, in = limit_periodic(i-1 - local_CFLz, nz), limit_periodic(i - local_CFLz, nz)
            else
                is, in = limit_periodic(i - local_CFLz, nz), limit_periodic(i+1 - local_CFLz, nz)
            end


            SL.u_max[I] = max(u_monot[is, jw, k], u_monot[in, jw, k], u_monot[is, je, k], u_monot[in, je, k])
            SL.u_min[I] = min(u_monot[is, jw, k], u_monot[in, jw, k], u_monot[is, je, k], u_monot[in, je, k])


            SL.u_monot[i,j,k] = min(max(SL.u_cubic[i,j,k], SL.u_min[I]), SL.u_max[I])
        end


    end

    u_monot .= SL.u_monot

end


function semi_lagrangian!(u_monot, SL, vc, vc_timestep, Δt, Grid, parameters, ϕ, ϕ0; method::Symbol=:quasi_monotone)

    @unpack nx, nz = Grid;
    @unpack grid_ad, Δx_ad, Δz_ad = parameters

    initial_pos_marker!(SL.x_t_depart, SL.z_t_depart, SL.v_t_half, vc, vc_timestep, Δt, Grid, parameters)

    if method == :quasi_monotone
        SL_quasi_monotone!(u_monot, SL, Δt, Grid, parameters)
    elseif method == :cubic
        SL_cubic!(u_monot, SL, Grid, parameters)
    elseif method == :linear
        SL_linear!(u_monot, SL, Grid, parameters)
    else
        error("Unknown method for the Semi-Lagrangian Scheme.")
    end

end
