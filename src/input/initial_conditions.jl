using Unitful
using Unitful: 𝐌, 𝐋, 𝐓, 𝚯
using Parameters

struct Grid{T1, T2, T3, T4}
    nx::T1
    nz::T1
    Lx::T2
    Lz::T2
    Δx::T2
    Δz::T2
    x::T3
    z::T3
    xs::T3
    zs::T3
    grid::T4
    tfinal::T2
    function Grid(nx::T1, nz::T1, Lx::T2, Lz::T2, tfinal::T2) where {T1 <: Integer, T2 <: Real}
        if Lx <= 0 || Lz <= 0
            error("Length should be positive.")
        elseif tfinal <= 0
            error("Total time should be positive.")
        else
            Δx = Lx / (nx)
            Δz = Lz / (nz)
            x = range(Δx/2, length=nx, stop= Lx-Δx/2)
            z = range(Δz/2, length=nz, stop= Lz-Δz/2)
            xs = range(0, length=nx+1, stop= Lx)
            zs = range(0, length=nz+1, stop= Lz)
            # x first, z second
            grid = (x=x' .* ones(nz), z=ones(nx)' .* z)

            T3 = typeof(x)
            T4 = typeof(grid)
            new{T1, T2, T3, T4}(nx, nz, Lx, Lz, Δx, Δz, x, z, xs, zs, grid, tfinal)
        end
    end
end

function Grid(;nx::Int, nz::Int, Lx::Unitful.Length, Lz::Unitful.Length, tfinal::Unitful.Time)
    Grid(nx, nz, ustrip(u"m", Lx), ustrip(u"m", Lz), ustrip(u"s", tfinal))
end


@with_kw struct Domain{T1 <: AbstractRange}
    x::T1
    z::T1
    # units::Dict
    visco_s::Float64 = 1e19  # in Pa.s
    visco_f::Array{Float64, 2} = ones(length(z), length(x)) .* 100  # in Pa.s
    T::Float64 = 1_200  # in C
    P::Float64 = 6  # in kbar
    ρ_s::Float64 = 2_700  # in kg.m⁻³
    ρ_f::Array{Float64, 2} = ones(length(z), length(x)) .* 2_200  # in kg.m⁻³
    R::Float64 = 0.01
    shear_mod::Float64 = 35e9  # in Pa
    ϕ::Array{Float64, 2} = Array{Float64}(undef, (length(z), length(x)))
    nb_element::Int = 9
    compo_f::Array{Float64, 3} = zeros(length(z), length(x), nb_element)
    compo_f_mol::Array{Float64, 1} = zeros(nb_element)
    compo_f_prev::Array{Float64, 3} = similar(compo_f)
    g::Float64=-9.80665  # gravity constant, m/s²
    ϕ0::Float64=1e-3  # background porosity
    kc::Float64=1e-7  # permeability constant, not background!!
    n::Float32=3  # exponent for porosity term in permeability
    m::Float32=1  # exponent for porosity term in bulk viscosity
    b::Float32=1  # exponent for porosity term in elasticy
end

@with_kw struct Model
    grid::Grid
    domain::Domain
    advection_algo::Union{UWScheme, WENOScheme, SemiLagrangianScheme, MICScheme}
    # advection_algo
    Courant::Float64 = 1.5
    ϕ_ini::Array{Float64, 2} = copy(domain.ϕ)
    Pe_ini::Array{Float64, 2} = zeros(grid.nz, grid.nx)
    ϕ::Array{Float64, 2} = zeros(grid.nz, grid.nx)
    Pe::Array{Float64, 2} = zeros(grid.nz, grid.nx)
    u0::Array{Float64, 3} = zeros(grid.nz, grid.nx, 2)
    du0::Array{Float64, 3} = similar(u0)
    q_f::NamedTuple{(:x, :z), Tuple{Matrix{Float64}, Matrix{Float64}}} = (x=zeros(grid.nz, grid.nx+1),z=zeros(grid.nz+1, grid.nx))
    v_f::NamedTuple{(:x, :z), Tuple{Matrix{Float64}, Matrix{Float64}}} = (x=zeros(grid.nz, grid.nx+1),z=zeros(grid.nz+1, grid.nx))
    v_s::NamedTuple{(:x, :z), Tuple{Matrix{Float64}, Matrix{Float64}}} = (x=zeros(grid.nz, grid.nx+1),z=zeros(grid.nz+1, grid.nx))
    vc_s::NamedTuple{(:x, :z), Tuple{Matrix{Float64}, Matrix{Float64}}} = (x=zeros(grid.nz, grid.nx),z=zeros(grid.nz, grid.nx))
    g::Float64=-9.80665  # gravity constant, m/s²
    ϕ0::Float64=1e-3  # background porosity
    kc::Float64=1e-7  # permeability constant, not background!!
    n::Float32=3  # exponent for porosity term in permeability
    m::Float32=1  # exponent for porosity term in bulk viscosity
    b::Float32=1  # exponent for porosity term in elasticy
    visco_f0::Float64 = (maximum(domain.visco_f) + minimum(domain.visco_f)) / 2
    ρ0::Float64 = domain.ρ_s - (maximum(domain.ρ_f) + minimum(domain.ρ_f)) / 2
    compaction_l::Float64 = sqrt(domain.visco_s * kc * ϕ0^(n-m) / visco_f0)
    compaction_l_R::Float64 = compaction_l * sqrt(domain.R)
    compaction_Pe::Float64 = ρ0 * abs(g) * compaction_l
    c0::Float64 = kc * ϕ0^n * ρ0 * abs(g) / (maximum(domain.visco_f) * ϕ0)
    compaction_t::Float64 = compaction_l / c0
    compaction_ϕ::Float64 = copy(ϕ0)
    De::Float64 = round(compaction_ϕ^(b-1) * (1 / domain.shear_mod) * compaction_Pe; digits=6)
    Δx_ad::Float64 = grid.Δx / compaction_l
    Δz_ad::Float64 = grid.Δz / compaction_l
    xs_ad::StepRangeLen = grid.xs ./ compaction_l
    xs_ad_vec::Array{Float64, 1} = collect(xs_ad)
    x_ad::StepRangeLen = grid.x ./ compaction_l
    x_ad_vec::Array{Float64, 1} = collect(x_ad)
    zs_ad::StepRangeLen = grid.zs ./ compaction_l
    zs_ad_vec::Array{Float64, 1} = collect(zs_ad)
    z_ad::StepRangeLen = grid.z ./ compaction_l
    z_ad_vec::Array{Float64, 1} = collect(z_ad)
    grid_ad::NamedTuple{(:x, :z), Tuple{Matrix{Float64}, Matrix{Float64}}} = (x=x_ad' .* ones(grid.nz), z=ones(grid.nx)' .* z_ad)
    t_count::Array{Float64, 1} = zeros(1)
    path_data::String = ""
end
