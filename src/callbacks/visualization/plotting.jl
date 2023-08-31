using Parameters
using Plots


# function to plot at each timestep
function plotting_tpf(u, t, integrator)
    @unpack x, z, Δx, Δz, tfinal = integrator.p[:grid]
    @unpack compaction_ϕ, compaction_Pe, compaction_t, c0, v_f, ϕ_ini, vc_s, v_s = integrator.p[:parameters]
    @unpack compo_f, visco_f = integrator.p[:domain]

    time_model = round((t * compaction_t / (3600 * 24 * 365.25 * 1e6)), sigdigits=2)

    l = @layout [a b c d]

    p1 = heatmap(x, z, log10.(u[:,:,2] .* compaction_ϕ),  xlim=(0, last(x)), ylim=(0, last(z)), title="ϕ")
    p2 = heatmap(x, z, u[:,:,1].* compaction_Pe,  xlim=(0, last(x)), ylim=(0, last(z)), title="Pe")
    p3 = heatmap(x, z, compo_f[:,:,1],  xlim=(0, last(x)), ylim=(0, last(z)), title="wt% of SiO2")
    p4 = heatmap(v_f[:z], title="Fluid velocity")

    display(plot(p1, p2, p3, p4, layout = l, plot_title="time=$(time_model) Ma"))

    # l = @layout [a b c]
    # p1 = heatmap(compo_f[:,40:60,1], title="wt% of SiO2")
    # p2 = heatmap(v_f[:z][:,40:60], title="vf Z")
    # p3 = heatmap(v_s[:z][:,40:60], title="vs Z")

    # display(plot(p1, p2, p3, layout=l))
end




