using Parameters
import Base.Threads.@threads

@with_kw struct WENOScheme
    nx::Int
    nz::Int
    u0::Array{Float64, 3}
    f::Array{Float64, 3} = similar(u0, Float64)
    fP::Array{Float64, 3} = similar(u0, Float64)  # positive flux
    fN::Array{Float64, 3} = similar(u0, Float64)  # negative flux
    fL::Array{Float64, 3} = similar(u0, Float64)
    fR::Array{Float64, 3} = similar(u0, Float64)
    fT::Array{Float64, 3} = similar(u0, Float64)
    fB::Array{Float64, 3} = similar(u0, Float64)
    r::Array{Float64, 3} = similar(u0, Float64)
    u_JS::Array{Float64, 3} = similar(u0, Float64)
    u_Z::Array{Float64, 3} = similar(u0, Float64)
    ut::Array{Float64, 3} = similar(u0, Float64)  # temporary array
    vc_f::NamedTuple{(:x, :z), Tuple{Matrix{Float64}, Matrix{Float64}}} = (x=zeros(nz, nx),z=zeros(nz, nx))
    algo_name::String = "WENO"
end


# u3 is the center
function WENO_u_upwind(u1, u2, u3, u4, u5, method)

    # Smoothness indicators
    β0 = 13/12 * (u1 - 2*u2 + u3)^2 + 1/4 * (u1 - 4*u2 + 3*u3)^2
    β1 = 13/12 * (u2 - 2*u3 + u4)^2 + 1/4 * (u2 - u4)^2
    β2 = 13/12 * (u3 - 2*u4 + u5)^2 + 1/4 * (3*u3 - 4*u4 + u5)^2

    # Borges et al. 2008 formulation
    if method == "Z"
        τ = abs(β0 - β2)
    end

    # Nonlinear weights w
    d0L = 1/10
    d1L = 3/5
    d2L = 3/10

    ϵ = 1e-6

    if method == "JS"
        α0L = d0L / (β0 + ϵ)^2
        α1L = d1L / (β1 + ϵ)^2
        α2L = d2L / (β2 + ϵ)^2

    # Borges et al. 2008 formulation
    elseif method == "Z"
        α0L = d0L * (1 + (τ / (β0 + ϵ))^2)
        α1L = d1L * (1 + (τ / (β1 + ϵ))^2)
        α2L = d2L * (1 + (τ / (β2 + ϵ))^2)

    else
        error("Unknown method for the WENO Scheme.")
    end

    w0L = α0L / (α0L + α1L + α2L)
    w1L = α1L / (α0L + α1L + α2L)
    w2L = α2L / (α0L + α1L + α2L)

    # Candidate stencils
    s0L = 1/3 * u1 - 7/6 * u2 + 11/6 * u3
    s1L = - 1/6 * u2 + 5/6 * u3 + 1/3 * u4
    s2L = 1/3 * u3 + 5/6 * u4 - 1/6 * u5

    # flux upwind
    fL = w0L * s0L + w1L * s1L + w2L * s2L

    return fL
end

# u3 is the center
function WENO_u_downwind(u1, u2, u3, u4, u5, method)

    # Smoothness indicators
    β0 = 13/12 * (u1 - 2*u2 + u3)^2 + 1/4 * (u1 - 4*u2 + 3*u3)^2
    β1 = 13/12 * (u2 - 2*u3 + u4)^2 + 1/4 * (u2 - u4)^2
    β2 = 13/12 * (u3 - 2*u4 + u5)^2 + 1/4 * (3*u3 - 4*u4 + u5)^2

    # Borges et al. 2008 formulation
    if method == "Z"
        τ = abs(β0 - β2)
    end

    # Nonlinear weights w
    d0R = 3/10
    d1R = 3/5
    d2R = 1/10

    ϵ = 1e-6

    # classical approach
    if method == "JS"

        α0R = d0R / (β0 + ϵ)^2
        α1R = d1R / (β1 + ϵ)^2
        α2R = d2R / (β2 + ϵ)^2

    # Borges et al. 2008 formulation
    elseif method == "Z"

        α0R = d0R * (1 + (τ / (β0 + ϵ))^2)
        α1R = d1R * (1 + (τ / (β1 + ϵ))^2)
        α2R = d2R * (1 + (τ / (β2 + ϵ))^2)

    else
        error("Unknown method for the WENO Scheme.")
    end

    w0R = α0R / (α0R + α1R + α2R)
    w1R = α1R / (α0R + α1R + α2R)
    w2R = α2R / (α0R + α1R + α2R)

    # Candidate stencils
    s0R = - 1/6 * u1 + 5/6 * u2 + 1/3 * u3
    s1R = 1/3 * u2 + 5/6 * u3 - 1/6 * u4
    s2R = 11/6 * u3 - 7/6 * u4 + 1/3 * u5

    # flux downwind
    fR = w0R * s0R + w1R * s1R + w2R * s2R

    return fR
end

function WENO_flux_upwind_x!(fL, u, nx, method)
    @inbounds @threads for I = CartesianIndices(u)
        i,j,k = Tuple(I)
        jw, jww = limit_periodic(j-1, nx), limit_periodic(j-2, nx)
        je, jee = limit_periodic(j+1, nx), limit_periodic(j+2, nx)

        u1 = u[i, jww, k]
        u2 = u[i, jw, k]
        u3 = u[i, j, k]
        u4 = u[i, je, k]
        u5 = u[i, jee, k]

        fL[I] = WENO_u_upwind(u1, u2, u3, u4, u5, method)
    end
end

function WENO_flux_upwind_z!(fB, u, nz, method)
    @inbounds @threads for I = CartesianIndices(u)
        i,j,k = Tuple(I)
        iw, iww = limit_periodic(i-1, nz), limit_periodic(i-2, nz)
        ie, iee = limit_periodic(i+1, nz), limit_periodic(i+2, nz)

        u1 = u[iww, j, k]
        u2 = u[iw, j, k]
        u3 = u[i, j, k]
        u4 = u[ie, j, k]
        u5 = u[iee, j, k]

        fB[I] = WENO_u_upwind(u1, u2, u3, u4, u5, method)
    end
end

function WENO_flux_downwind_x!(fR, u, nx, method)
    @inbounds @threads for I = CartesianIndices(u)
        i,j,k = Tuple(I)
        jw, jww = limit_periodic(j-1, nx), limit_periodic(j-2, nx)
        je, jee = limit_periodic(j+1, nx), limit_periodic(j+2, nx)

        u1 = u[i, jww, k]
        u2 = u[i, jw, k]
        u3 = u[i, j, k]
        u4 = u[i, je, k]
        u5 = u[i, jee, k]

        fR[I] = WENO_u_downwind(u1, u2, u3, u4, u5, method)
    end
end

function WENO_flux_downwind_z!(fT, u, nz, method)
    @inbounds @threads for I = CartesianIndices(u)
        i,j,k = Tuple(I)
        iw, iww = limit_periodic(i-1, nz), limit_periodic(i-2, nz)
        ie, iee = limit_periodic(i+1, nz), limit_periodic(i+2, nz)

        u1 = u[iww, j, k]
        u2 = u[iw, j, k]
        u3 = u[i, j, k]
        u4 = u[ie, j, k]
        u5 = u[iee, j, k]

        fT[I] = WENO_u_downwind(u1, u2, u3, u4, u5, method)
    end
end

function rhs!(u, vx, vz, WENO, grid, parameters, method)

    @unpack nx, nz = grid
    @unpack Δx_ad, Δz_ad  = parameters
    @unpack fL, fR, fB, fT, r = WENO

    WENO_flux_upwind_x!(fL, u, nx, method)
    WENO_flux_downwind_x!(fR, u, nx, method)

    WENO_flux_upwind_z!(fB, u, nz, method)
    WENO_flux_downwind_z!(fT, u, nz, method)

    @inbounds @threads for I = CartesianIndices(u)
        i,j,k = Tuple(I)

        jw, je = limit_periodic(j-1, nx), limit_periodic(j+1, nx)
        is, in = limit_periodic(i-1, nz), limit_periodic(i+1, nz)

        r[I] = max(vx[i,j], 0) * (fL[i, j, k] - fL[i, jw, k]) / Δx_ad +
               min(vx[i,j], 0) * (fR[i, je, k] - fR[i, j, k]) / Δx_ad +
               max(vz[i,j], 0) * (fB[i, j, k] - fB[is, j, k]) / Δz_ad +
               min(vz[i,j], 0) * (fT[in, j, k] - fT[i, j, k]) / Δz_ad
    end
end



function WENO_scheme!(u, v, WENO, grid, parameters, Δt; method="JS")

    @unpack ut, r = WENO

    rhs!(u, v[:x], v[:z], WENO, grid, parameters, method)

    @inbounds @threads for I = CartesianIndices(u)
        ut[I] = u[I] - Δt * r[I]
    end

    rhs!(ut, v[:x], v[:z], WENO, grid, parameters, method)

    @inbounds @threads for I = CartesianIndices(u)
        ut[I] = 0.75 * u[I] + 0.25 * ut[I] - 0.25 * Δt * r[I]
    end

    rhs!(ut, v[:x], v[:z], WENO, grid, parameters, method)

    @inbounds @threads for I = CartesianIndices(u)
        u[I] = 1.0/3.0 * u[I] + 2.0/3.0 * ut[I] - (2.0/3.0) * Δt * r[I]
    end

end