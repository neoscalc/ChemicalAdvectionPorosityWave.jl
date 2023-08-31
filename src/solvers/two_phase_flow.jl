
#=
Equations dimensionless:
        ∂ϕ/∂t = - ∂/∂z( ϕ^n (∂/∂z(Pe) + 1) )
     De ∂Pe/∂t =  ∂/∂z( ϕ^n (∂/∂z(Pe) + 1) ) - ϕ^m Pe

discretisation:
        ∂ϕ/∂t = - 1/Δx * ((ϕ_i+1/2)^n * (((Pe_i+1 - Pe_i) / Δx) + 1)
                          - (ϕ_i-1/2)^n * (((Pe_i - Pe_i-1) / Δx) + 1))
        ∂Pe/∂t = 1/De * (1/Δx * ((ϕ_i+1/2)^n * (((Pe_i+1 - Pe_i) / Δx) + 1)
                                 - (ϕ_i-1/2)^n * (((Pe_i - Pe_i-1) / Δx) + 1))
                 - ϕ_i^m * Pe_i)

=#

using Parameters
import Base.@propagate_inbounds

# Define the problem in form of ODEs by discretizing in space
function porosity_wave(du, u, p, t)

    Pe = @view u[:, :, 1]
    ϕ = @view u[:, :, 2]
    @unpack nz, nx = p.grid
    @unpack Δz_ad, Δx_ad, n, m, b, De, visco_f0, ρ0 = p.parameters
    @unpack R, visco_f, ρ_s, ρ_f, g = p.domain

    @propagate_inbounds @inline av_k_z(i,in,j) = 0.5*(ϕ[i,j]^n+ϕ[in,j]^n)
    @propagate_inbounds @inline av_k_x(i,j,je) = 0.5*(ϕ[i,j]^n+ϕ[i,je]^n)
    @propagate_inbounds @inline av_μf_z(i,in,j) = 0.5*(visco_f[i,j]+visco_f[in,j]) / visco_f0
    @propagate_inbounds @inline av_μf_x(i,j,je) = 0.5*(visco_f[i,j]+visco_f[i,je]) / visco_f0
    @propagate_inbounds @inline av_Δρ_z(i,in,j) = (ρ_s - 0.5*(ρ_f[i,j] + ρ_f[in,j])) * g / (ρ0*g)
    @propagate_inbounds @inline qz(i,in,j) = av_k_z(i,in,j) / av_μf_z(i,in,j) * ((Pe[in,j]-Pe[i,j]) / Δz_ad + av_Δρ_z(i,in,j))
    @propagate_inbounds @inline qx(i,j,je) = av_k_x(i,j,je) / av_μf_x(i,j,je) * (Pe[i,je]-Pe[i,j]) / Δx_ad

    # Approximation of the Heaviside function copied from here: https://discourse.julialang.org/t/handling-instability-when-solving-ode-problems/9019/5
    v1 = 0.0  # min of the Heaviside function
    v2 = 1.0  # max of the Heaviside function
    @propagate_inbounds @inline H(Pe) = (v2-v1)*tanh(Pe / Δz_ad )/2 + (v1 + v2) / 2

    # loop over all the points
    @inbounds for I in CartesianIndices((nz, nx))
        i, j = Tuple(I)
        is, in = limit_periodic(i-1, nz), limit_periodic(i+1, nz)
        jw, je = limit_periodic(j-1, nx), limit_periodic(j+1, nx)

        # pressure
        du[i, j, 1] = 1 /(ϕ[i, j]^b * De) *
            ((qx(i,j,je) - qx(i,jw,j)) / Δx_ad +
             (qz(i,in,j) - qz(is,i,j)) / Δz_ad -
             ϕ[i, j]^m * Pe[i, j] / (R - H(Pe[i, j]) * (R - 1)))

        # porosity
        du[i, j, 2] = - (ϕ[i, j]^b * De * du[i, j, 1] + ϕ[i, j]^m * Pe[i, j] / (R - H(Pe[i, j]) * (R - 1)))
    end
end
