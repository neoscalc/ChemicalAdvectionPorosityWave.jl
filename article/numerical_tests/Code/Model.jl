module Model

using Parameters
using Base.Threads: Atomic

@with_kw mutable struct ModelParameters
    u0::Array{Float64, 2}
    Lx::Float64
    Ly::Float64
    Δx::Float64
    Δy::Float64
    nx::Int = round(Int,Lx/Δx)
    ny::Int = round(Int,Ly/Δy)
    x::StepRangeLen = range(Δx/2, length=nx, stop= Lx-Δx/2)
    y::StepRangeLen = range(Δy/2, length=ny, stop= Ly-Δy/2)
    tmax::Float64
    # x first, y second
    grid::Tuple{Matrix{Float64}, Matrix{Float64}} = (x' .* ones(ny), ones(nx)' .* y)
    vx0::Array{Float64, 2} = ones((ny,nx)) .* Lx
    vy0::Array{Float64, 2} = ones((ny,nx)) .* Ly
    v0::Tuple{Matrix{Float64}, Matrix{Float64}} = (vx0, vy0)
    Δt::Float64
end

@with_kw struct UWScheme
    nx::Int
    ny::Int
    u0::Array{Float64, 2}
    u_old::Array{Float64, 2} = copy(u0)  # temporary array
end

@with_kw struct SemiLagrangianScheme
    nx::Int
    ny::Int
    u0::Array{Float64, 2}
    u::Array{Float64, 2} = copy(u0)
    u_monot::Array{Float64, 2} = zeros(ny, nx)
    u_cubic::Array{Float64, 2} = zeros(ny, nx)
    u_linear::Array{Float64, 2} = zeros(ny, nx)
    u_max::Array{Float64, 2} = zeros(ny, nx)
    u_min::Array{Float64, 2} = zeros(ny, nx)
    x_t0::Array{Float64, 2} = zeros(ny, nx)
    y_t0::Array{Float64, 2} = zeros(ny, nx)
    x_t_depart::Array{Float64, 2} = zeros(ny, nx)
    y_t_depart::Array{Float64, 2} = zeros(ny, nx)
    w::Array{Float64, 2} = zeros(ny, nx)
    v_t_half::Tuple{Matrix{Float64}, Matrix{Float64}} = (zeros(ny, nx),zeros(ny, nx))
    v_timestep::Tuple{Matrix{Float64}, Matrix{Float64}} = (zeros(ny, nx),zeros(ny, nx))
end

@with_kw struct WENOScheme
    nx::Int
    ny::Int
    u0::Array{Float64, 2}
    f::Array{Float64, 2} = zeros(ny, nx)
    fL::Array{Float64, 2} = zeros(ny, nx)
    fR::Array{Float64, 2} = zeros(ny, nx)
    fT::Array{Float64, 2} = zeros(ny, nx)
    fB::Array{Float64, 2} = zeros(ny, nx)
    r::Array{Float64, 2} = zeros(ny, nx)
    ut::Array{Float64, 2} = similar(u0, Float64)  # temporary array
end

@with_kw struct MICScheme
    u0::Array{Float64, 2}
    nx::Int
    ny::Int
    Lx::Float64
    Ly::Float64
    Δx::Float64 = Lx/nx
    Δy::Float64 = Ly/ny
    nx_marker::Int = nx*3
    ny_marker::Int = ny*3
    x::StepRangeLen = range(Δx/2, length=nx, stop= Lx-Δx/2)
    x_vec::Array{Float64, 1} = collect(x)
    y::StepRangeLen = range(Δy/2, length=ny, stop= Ly-Δy/2)
    y_vec::Array{Float64, 1} = collect(y)
    X::Array{Float64, 2} = x' .* ones(length(y))
    Y::Array{Float64, 2} = ones(length(x))' .* y
    x_mark::Array{Float64, 1} = collect(LinRange(Δx/2, Lx-Δx/2, nx_marker))
    y_mark::Array{Float64, 1} = collect(LinRange(Δy/2, Ly-Δy/2, ny_marker))
    X_mark::Array{Float64, 1} = (x_mark' .* ones(length(y_mark)))[:]
    Y_mark::Array{Float64, 1} = (ones(length(x_mark))' .* y_mark)[:]
    X_mark_save::Array{Float64, 1} = similar(X_mark)
    Y_mark_save::Array{Float64, 1} = similar(Y_mark)
    u_mark::Array{Float64, 1} = similar(X_mark)  # temporary array
    v_t_old::Tuple{Vector{Float64}, Vector{Float64}} = (similar(X_mark),similar(Y_mark))
    v_timestep::Tuple{Vector{Float64}, Vector{Float64}} = (similar(X_mark),similar(Y_mark))
    u_sum::Array{Float64, 2} = similar(u0)
    u_sum_atomic::Matrix{Atomic{Float64}} = [Atomic{Float64}(0.0) for _ in 1:size(u_sum,1), _ in 1:size(u_sum,2)]  # Assuming nx and nz are the dimensions of your 2D grid
    wt_sum::Array{Float64, 2} = similar(u0)
    wt_sum_atomic::Matrix{Atomic{Float64}} = [Atomic{Float64}(0.0) for _ in 1:size(wt_sum,1), _ in 1:size(wt_sum,2)]
end


end