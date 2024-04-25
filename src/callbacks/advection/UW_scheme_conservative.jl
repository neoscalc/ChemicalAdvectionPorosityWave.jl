using Parameters
import Base.Threads.@threads

@with_kw struct UWScheme
    nx::Int
    nz::Int
    sum_element::Array{Float64, 2} = zeros(nz, nx)
    u_old::Array{Float64, 2} = zeros(nz, nx)  # temporary array
    vc_f::NamedTuple{(:x, :z), Tuple{Matrix{Float64}, Matrix{Float64}}} = (x=zeros(nz, nx),z=zeros(nz, nx))
    algo_name::String = "Upwind"
end


function upwind_scheme!(compo_f, compo_f_prev, v_c, Δt, grid, parameters)

    @unpack nx, nz = grid;
    @unpack Δx_ad, Δz_ad = parameters

    compo_f_prev .= compo_f



    @inbounds @threads for I = CartesianIndices(compo_f)
        i, j, k = Tuple(I)
        is,in = limit_periodic(i-1, nz), limit_periodic(i+1, nz)
        jw, je = limit_periodic(j-1, nx), limit_periodic(j+1, nx)

        # solve advection equation using an upwind finite difference discretisation in 2D
        # first check the sign of the velocity in the x and z direction and then apply the appropriate finite difference scheme

        if v_c[:x][i,j] > 0
            if v_c[:z][i,j] > 0
                compo_f[I] = compo_f_prev[I] - (Δt / Δz_ad) * (v_c[:z][i,j] * compo_f_prev[I] - v_c[:z][is,j]*compo_f_prev[is,j,k]) - (Δt / Δx_ad) * (v_c[:x][i,j] * compo_f_prev[I] - v_c[:x][i,jw] * compo_f_prev[i,jw,k])
            else v_c[:z][i,j] < 0
                compo_f[I] = compo_f_prev[I] - (Δt / Δz_ad) * (v_c[:z][in,j] * compo_f_prev[in,j,k] - v_c[:z][i,j] * compo_f_prev[I]) - (Δt / Δx_ad) * (v_c[:x][i,j] * compo_f_prev[I] - v_c[:x][i,jw] * compo_f_prev[i,jw,k])
            end
        else
            if v_c[:z][i,j] > 0
                compo_f[I] = compo_f_prev[I] - (Δt / Δz_ad) * (v_c[:z][i,j] * compo_f_prev[I] - v_c[:z][is,j] * compo_f_prev[is,j,k]) - (Δt / Δx_ad) * (v_c[:x][i,je] * compo_f_prev[i,je,k] - v_c[:x][i,j]* compo_f_prev[I])
            else v_c[:z][i,j] < 0
                compo_f[I] = compo_f_prev[I] - (Δt / Δz_ad) * (v_c[:z][in,j] * compo_f_prev[in,j,k] - v_c[:z][i,j] * compo_f_prev[I]) - (Δt / Δx_ad) * (v_c[:x][i,je] * compo_f_prev[i,je,k] - v_c[:x][i,j] * compo_f_prev[I])
            end
        end

        # add source terms
        # compo_f[I] += Δt / Δz_ad * compo_f_prev[I] * (v_c[:z][i,j] - v_c[:z][is,j]) + Δt / Δx_ad * compo_f_prev[I] * (v_c[:x][i,j] - v_c[:x][i,jw])

        # add source term as upwind
        # if compo_f_prev[I] > 0
        #     compo_f[I] += Δt / Δz_ad * compo_f_prev[I] * (v_c[:z][i,j] - v_c[:z][is,j]) + Δt / Δx_ad * compo_f_prev[I] * (v_c[:x][i,j] - v_c[:x][i,jw])
        # else
        #     compo_f_prev[I] += Δt / Δz_ad * compo_f_prev[I] * (v_c[:z][in,j] - v_c[:z][i,j]) + Δt / Δx_ad * compo_f_prev[I] * (v_c[:x][i,je] - v_c[:x][i,j])
        # end
        # compo_f[I] = compo_f_prev[I] -
        #             (Δt / Δz) * (max(0,v_f[:z][i, j]) * (compo_f_prev[I] - compo_f_prev[is,j,k]) + min(0,v_f[:z][i, j]) * (compo_f_prev[in,j,k] - compo_f_prev[I])) -
        #             (Δt / Δx) * (max(0,v_f[:x][i, j]) * (compo_f_prev[I] - compo_f_prev[i,jw,k]) + min(0,v_f[:x][i, j]) * (compo_f_prev[i,je,k] - compo_f_prev[I]))
    end

end

