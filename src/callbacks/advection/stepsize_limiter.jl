using Parameters

function dtmaxC(u,p,t)
    @unpack tfinal = p[:grid]
    @unpack v_f, compaction_t, Courant, Δx_ad, Δz_ad  = p[:parameters]

    # addimensional maximum timestep from Courant number formula in 2D
    tmax = Courant * min(Δx_ad / maximum(v_f[:x]), Δz_ad / maximum(v_f[:z]))

    if isnan(tmax)
        tmax = tfinal / compaction_t
    end

    return tmax
end

