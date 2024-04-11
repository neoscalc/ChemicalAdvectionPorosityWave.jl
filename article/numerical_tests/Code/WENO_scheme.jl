module WENO_scheme

import Base.Threads.@threads
using Parameters

@inline function limit_periodic(a, n)
    # check if index is on the boundary, if yes take value on the opposite for periodic, if not, don't change the value
    a > n ? a = a-n : a < 1 ? a = a+n : a = a
end


# u3 is the center
function WENO_u_upwind(u1, u2, u3, u4, u5, method)

    # Smoothness indicators
    β0 = 13/12 * (u1 - 2*u2 + u3)^2 + 1/4 * (u1 - 4*u2 + 3*u3)^2
    β1 = 13/12 * (u2 - 2*u3 + u4)^2 + 1/4 * (u2 - u4)^2
    β2 = 13/12 * (u3 - 2*u4 + u5)^2 + 1/4 * (3*u3 - 4*u4 + u5)^2

    # Borges et al. 2008 formulation
    if method == :Z
        τ = abs(β0 - β2)
    end

    # Nonlinear weights w
    d0L = 1/10
    d1L = 3/5
    d2L = 3/10

    ϵ = 1e-6

    if method == :JS
        α0L = d0L / (β0 + ϵ)^2
        α1L = d1L / (β1 + ϵ)^2
        α2L = d2L / (β2 + ϵ)^2

    # Borges et al. 2008 formulation
    elseif method == :Z
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
    if method == :Z
        τ = abs(β0 - β2)
    end

    # Nonlinear weights w
    d0R = 3/10
    d1R = 3/5
    d2R = 1/10

    ϵ = 1e-6

    # classical approach
    if method == :JS

        α0R = d0R / (β0 + ϵ)^2
        α1R = d1R / (β1 + ϵ)^2
        α2R = d2R / (β2 + ϵ)^2

    # Borges et al. 2008 formulation
    elseif method == :Z

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


function WENO_flux_upwind_x!(fL, u, nx, ny, method)
    @inbounds for I = CartesianIndices((ny, nx))
        i,j = Tuple(I)
        jw, jww = limit_periodic(j-1, nx), limit_periodic(j-2, nx)
        je, jee = limit_periodic(j+1, nx), limit_periodic(j+2, nx)

        u1 = u[i, jww]
        u2 = u[i, jw]
        u3 = u[i, j]
        u4 = u[i, je]
        u5 = u[i, jee]

        fL[I] = WENO_u_upwind(u1, u2, u3, u4, u5, method)
    end
end


function WENO_flux_upwind_y!(fB, u, nx, ny, method)
    @inbounds @threads for I = CartesianIndices((ny, nx))
        i,j = Tuple(I)
        iw, iww = limit_periodic(i-1, ny), limit_periodic(i-2, ny)
        ie, iee = limit_periodic(i+1, ny), limit_periodic(i+2, ny)

        u1 = u[iww, j]
        u2 = u[iw, j]
        u3 = u[i, j]
        u4 = u[ie, j]
        u5 = u[iee, j]

        fB[I] = WENO_u_upwind(u1, u2, u3, u4, u5, method)
    end
end

function WENO_flux_downwind_x!(fR, u, nx, ny, method)
    @inbounds @threads for I = CartesianIndices((ny, nx))
        i,j = Tuple(I)
        jw, jww = limit_periodic(j-1, nx), limit_periodic(j-2, nx)
        je, jee = limit_periodic(j+1, nx), limit_periodic(j+2, nx)

        u1 = u[i, jww]
        u2 = u[i, jw]
        u3 = u[i, j]
        u4 = u[i, je]
        u5 = u[i, jee]

        fR[I] = WENO_u_downwind(u1, u2, u3, u4, u5, method)
    end
end

function WENO_flux_downwind_y!(fT, u, nx, ny, method)
    @inbounds @threads for I = CartesianIndices((ny, nx))
        i,j = Tuple(I)
        iw, iww = limit_periodic(i-1, ny), limit_periodic(i-2, ny)
        ie, iee = limit_periodic(i+1, ny), limit_periodic(i+2, ny)

        u1 = u[iww, j]
        u2 = u[iw, j]
        u3 = u[i, j]
        u4 = u[ie, j]
        u5 = u[iee, j]

        fT[i,j] = WENO_u_downwind(u1, u2, u3, u4, u5, method)
    end
end


function rhs!(u, vx, vy, WENO, parameters, method)

    @unpack Δx, Δy, nx, ny = parameters
    @unpack fL, fR, fB, fT, r = WENO

    WENO_flux_upwind_x!(fL, u, nx, ny, method)
    WENO_flux_downwind_x!(fR, u, nx, ny, method)

    WENO_flux_upwind_y!(fB, u, nx, ny, method)
    WENO_flux_downwind_y!(fT, u, nx, ny, method)

    @inbounds @threads for I = CartesianIndices((ny, nx))
        i,j = Tuple(I)

        jw, je = limit_periodic(j-1, nx), limit_periodic(j+1, nx)
        is, in = limit_periodic(i-1, ny), limit_periodic(i+1, ny)

        r[I] = max(vx[I], 0) * (fL[i, j] - fL[i, jw]) / Δx +
               min(vx[I], 0) * (fR[i, je] - fR[i, j]) / Δx +
               max(vy[I], 0) * (fB[i, j] - fB[is, j]) / Δy +
               min(vy[I], 0) * (fT[in, j] - fT[i, j]) / Δy
    end
end


function WENO_scheme!(u, v, WENO, parameters; method=:Z)

    @unpack Δx, Δy, nx, ny, Δt = parameters
    @unpack ut, r = WENO

    rhs!(u, v[1], v[2], WENO, parameters, method)

    @inbounds @threads for I = CartesianIndices((ny, nx))
        ut[I] = u[I] - Δt * r[I]
    end

    rhs!(ut, v[1], v[2], WENO, parameters, method)

    @inbounds @threads for I = CartesianIndices((ny, nx))
        ut[I] = 0.75 * u[I] + 0.25 * ut[I] - 0.25 * Δt * r[I]
    end

    rhs!(ut, v[1], v[2], WENO, parameters, method)

    @inbounds @threads for I = CartesianIndices((ny, nx))
        u[I] = 1.0/3.0 * u[I] + 2.0/3.0 * ut[I] - (2.0/3.0) * Δt * r[I]
    end

end

end