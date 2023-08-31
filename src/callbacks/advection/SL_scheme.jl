using Parameters
using Interpolations
using Plots
import Base.Threads.@threads

@with_kw struct SemiLagrangianScheme
    nx::Int
    nz::Int
    u0::Array{Float64, 3}
    u_monot::Array{Float64, 3} = zeros(nz, nx, size(u0,3))
    u_cubic::Array{Float64, 3} = zeros(nz, nx, size(u0,3))
    u_linear::Array{Float64, 3} = zeros(nz, nx, size(u0,3))
    u_max::Array{Float64, 2} = zeros(nz, nx)
    u_min::Array{Float64, 2} = zeros(nz, nx)
    x_t0::Array{Float64, 2} = zeros(nz, nx)
    z_t0::Array{Float64, 2} = zeros(nz, nx)
    x_t_half::Array{Float64, 2} = zeros(nz, nx)
    z_t_half::Array{Float64, 2} = zeros(nz, nx)
    w::Array{Float64, 2} = zeros(nz, nx)
    v_t_half::NamedTuple{(:x, :z), Tuple{Matrix{Float64}, Matrix{Float64}}} = (x=zeros(nz, nx),z=zeros(nz, nx))
    vc_f::NamedTuple{(:x, :z), Tuple{Matrix{Float64}, Matrix{Float64}}} = (x=zeros(nz, nx),z=zeros(nz, nx))
    vc_previous::NamedTuple{(:x, :z), Tuple{Matrix{Float64}, Matrix{Float64}}} = (x=zeros(nz, nx),z=zeros(nz, nx))
    div_v::Array{Float64, 2} = zeros(nz, nx)
    det::Array{Float64, 2} = zeros(nz, nx)
    algo_name::String = "Semi-Lagrangian"
end

# second order implicit mid point method
function interpol_velocity_SL!(v_t_half, v, v_timestep, SL, Δt, Grid, parameters)

    @unpack nx, nz = Grid;
    @unpack grid_ad, x_ad, z_ad = parameters;
    @unpack x_t_half, z_t_half = SL;

    x_t_half .= grid_ad[:x] .- v[:x] .* Δt/2  # calculate the previous position of the particle at position t+1/2 from the position at t+1
    z_t_half .= grid_ad[:z] .- v[:z] .* Δt/2  # calculate the previous position of the particle at position t+1/2 from the position at t+1

    # for the 1st timestep, when v_timestep = 0
    if !iszero(v_timestep[1]) && !iszero(v_timestep[2])
    # interpolation
        # itp_x = interpolate(((v[:x] .+ v_timestep[:x]) / 2), BSpline(Linear()))
        # itp_z = interpolate(((v[:z] .+ v_timestep[:z]) / 2), BSpline(Linear()))
        setp_x = extrapolate(scale(interpolate(v[:x], BSpline(Linear())), z_ad, x_ad), Line())
        setp_z = extrapolate(scale(interpolate(v[:z], BSpline(Linear())), z_ad, x_ad), Line())
    else
        # itp_x = interpolate(v[:x], BSpline(Linear()))
        # itp_z = interpolate(v[:z], BSpline(Linear()))
        setp_x = extrapolate(scale(interpolate(v[:x], BSpline(Linear())), z_ad, x_ad), Line())
        setp_z = extrapolate(scale(interpolate(v[:z], BSpline(Linear())), z_ad, x_ad), Line())
    end

    # scaled interpolation
    # sitp = scale(itp_x, z_ad, x_ad)
    # setp = extrapolate(sitp, Line())
    # # scaled interpolation with extrapolation on the boundaries

    @inbounds @threads for I in CartesianIndices((nz, nx))
        v_t_half[:x][I] = setp_x(grid_ad[:z][I],x_t_half[I]);
    end

    # scaled interpolation
    # sitp = scale(itp_z, z_ad, x_ad)
    # setp = extrapolate(sitp, Line())
    # # scaled interpolation with extrapolation on the boundaries

    @inbounds @threads for I in CartesianIndices((nz, nx))
        v_t_half[:z][I] = setp_z(z_t_half[I], grid_ad[:x][I],);
    end

    # iterate 3 times to improve approximation
    for _ in 1:3
        SL.x_t_half .= grid_ad[:x] .- v_t_half[:x] .* Δt/2  # calculate the previous position of the particle at position t+1/2 from the position at t+1
        SL.z_t_half .= grid_ad[:z] .- v_t_half[:z] .* Δt/2  # calculate the previous position of the particle at position t+1/2 from the position at t+1

        # itp_x = interpolate(v_t_half[:x], BSpline(Linear()))
        # itp_z = interpolate(v_t_half[:z], BSpline(Linear()))

        # # scaled interpolation
        # sitp = scale(itp_x, z_ad, x_ad)
        # setp = extrapolate(sitp, Line())
        # # scaled interpolation with extrapolation on the boundaries
        setp_x = extrapolate(scale(interpolate(v_t_half[:x], BSpline(Linear())), z_ad, x_ad), Line())

        @inbounds @threads for I in CartesianIndices((nz, nx))
            v_t_half[:x][I] = setp_x(grid_ad[:z][I],SL.x_t_half[I]);
        end

        # scaled interpolation
        # sitp = scale(itp_z, z_ad, x_ad)
        # setp = extrapolate(sitp, Line())
        # # scaled interpolation with extrapolation on the boundaries
        setp_z = extrapolate(scale(interpolate(v_t_half[:z], BSpline(Linear())), z_ad, x_ad), Line())

        @inbounds @threads for I in CartesianIndices((nz, nx))
            v_t_half[:z][I] = setp_z(z_t_half[I], grid_ad[:x][I]);
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

        @inbounds @threads for I = CartesianIndices(SL.x_t0)
            i,j = Tuple(I)
            SL.u_cubic[i,j,k] = setp(SL.z_t0[i,j], SL.x_t0[i,j])
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


            SL.u_max[i, j] = max(u_monot[is, jw, k], u_monot[in, jw, k], u_monot[is, je, k], u_monot[in, je, k])
            SL.u_min[i, j] = min(u_monot[is, jw, k], u_monot[in, jw, k], u_monot[is, je, k], u_monot[in, je, k])


            SL.u_monot[i,j,k] = min(max(SL.u_cubic[i,j,k], SL.u_min[i, j]), SL.u_max[i, j])
        end


    end

    u_monot .= SL.u_monot

end

# function divergence_velocity!(div_v, v, Grid, Parameters)

#     @unpack nx, nz, Δx, Δz = Grid;
#     @unpack Δx_ad, Δz_ad, c0 = Parameters;

#     for I in CartesianIndices((nz,nx))
#         i,j = Tuple(I)
#         iw, ie = limit_neumann(i-1, nz), limit_neumann(i+1, nz)
#         js, jn = limit_neumann(j-1, nx), limit_neumann(j+1, nx)
#         div_v[I] = ((v[:x][i,jn] + v[:x][i,j]) - (v[:x][i,js] + v[:x][i,j])) * 0.5 / Δx_ad + ((v[:z][ie,j] + v[:z][i,j]) - (v[:z][iw,j] + v[:z][i,j])) * 0.5 / Δz_ad
#     end

#     @show maximum(div_v)

#     display(heatmap(div_v))

# end

# function determinant_jac!(det, div_v, Δt)
#     det .= exp.(.- Δt .* div_v)
# end

# # interpolate the jacobian's determinant
# function interpol_jac!(det_interp, det, SL, Grid, parameters)

#     @unpack nx, nz = Grid;
#     @unpack x_ad, z_ad = parameters
#     @unpack x_t0, z_t0 = SL

#     # interpolation Det of Jacobian
#     # scaled interpolation with extrapolation on the boundaries
#     setp = extrapolate(scale(interpolate(det, BSpline(Cubic(Periodic(OnCell())))), z_ad, x_ad), Periodic())

#     for I = CartesianIndices((nz, nx))
#         det_interp[I] = setp(z_t0[I],x_t0[I])
#     end

# end

function semi_lagrangian!(u_monot, SL, vc, vc_timestep, v, Δt, Grid, parameters, ϕ, ϕ0; method::String="linear")

    @unpack nx, nz = Grid;
    @unpack grid_ad, Δx_ad, Δz_ad = parameters

    interpol_velocity_SL!(SL.v_t_half, vc, vc_timestep, SL, Δt, Grid, parameters)

    SL.x_t0 .= grid_ad[:x] .- SL.v_t_half[:x] .* Δt  # calculate the previous position of the particle at position t from the position at t+1
    SL.z_t0 .= grid_ad[:z] .- SL.v_t_half[:z] .* Δt  # calculate the previous position of the particle at position t from the position at t+1

    if method == "quasi-monotone"
        # divergence_velocity!(SL.div_v, SL.v_t_half, Grid, parameters)
        # determinant_jac!(SL.det, SL.div_v, Δt)
        # interpol_jac!(SL.det_interp, SL.det, SL, Grid, Properties)
        SL_quasi_monotone!(u_monot, SL, Δt, Grid, parameters)
    elseif method == "cubic"
        SL_cubic!(u_monot, SL, Grid, parameters)
    elseif method == "linear"
        SL_linear!(u_monot, SL, Grid, parameters)
    else
        error("Unknown method for the Semi-Lagrangian Scheme.")
    end

end
