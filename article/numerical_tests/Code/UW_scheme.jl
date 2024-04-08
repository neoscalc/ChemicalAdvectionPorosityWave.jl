module UW_scheme


using Parameters

@inline function limit_periodic(a, n)
    # check if index is on the boundary, if yes take value on the opposite for periodic, if not, don't change the value
    a > n ? a = a-n : a < 1 ? a = a+n : a = a
end


function Upwind!(u_new, u_old, v, Δt, Param)

    @unpack nx, ny, Δx, Δy = Param;

    for I = CartesianIndices((ny, nx))
        i,j = Tuple(I)
        is,in = limit_periodic(i-1, ny), limit_periodic(i+1, ny)
        jw, je = limit_periodic(j-1, nx), limit_periodic(j+1, nx)

        if v[1][I] > 0
            if v[2][i,j] > 0
                u_new[I] = u_old[I] - (Δt / Δy) * v[2][i,j] * (u_old[I] - u_old[is,j]) - (Δt / Δx) * v[1][i,j] * (u_old[I] - u_old[i,jw])
            else v[2][i,j] < 0
                u_new[I] = u_old[I] - (Δt / Δy) * v[2][i,j] * (u_old[in,j] - u_old[I]) - (Δt / Δx) * v[1][i,j] * (u_old[I] - u_old[i,jw])
            end
        else
            if v[2][i,j] > 0
                u_new[I] = u_old[I] - (Δt / Δy) * v[2][i,j] * (u_old[I] - u_old[is,j]) - (Δt / Δx) * v[1][i,j] * (u_old[i,je] - u_old[I])
            else v[2][i,j] < 0
                u_new[I] = u_old[I] - (Δt / Δy) * v[2][i,j] * (u_old[in,j] - u_old[I]) - (Δt / Δx) * v[1][i,j] * (u_old[i,je] - u_old[I])
            end
        end


    end

    u_old .= u_new

end

end
