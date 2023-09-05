using Parameters
using Plots


# function to plot at each timestep
function plotting_tpf(u, t, integrator)
    @unpack x, z, xs, zs, Δx, Δz, tfinal = integrator.p[:grid]
    @unpack compaction_ϕ, compaction_Pe, compaction_t, c0, v_f, ϕ_ini, vc_s, v_s = integrator.p[:parameters]
    @unpack compo_f, visco_f = integrator.p[:domain]

    time_model = round((t * compaction_t / (3600 * 24 * 365.25 * 1e6)), sigdigits=2)

    l = @layout [a b ; c d]

    Plots.scalefontsizes(0.5)

    p1 = heatmap(x*1e-3, z*1e-3, log10.(u[:,:,2] .* compaction_ϕ),  xlim=(0, last(x)*1e-3), ylim=(0, last(z)*1e-3), title="log(Porosity)", ylabel= "Distance (km)")
    p2 = heatmap(x*1e-3, z*1e-3, u[:,:,1].* compaction_Pe .* 1e-5,  xlim=(0, last(x)*1e-3), ylim=(0, last(z)*1e-3), title="Effective Pressure (bar)")
    p3 = heatmap(x*1e-3, z*1e-3, compo_f[:,:,1],  xlim=(0, last(x)*1e-3), ylim=(0, last(z)*1e-3), title="wt% of SiO2", ylabel="Distance (km)",xlabel="Distance (km)")
    p4 = heatmap(x*1e-3, zs*1e-3, v_f[:z],  xlim=(0, last(x)*1e-3), ylim=(0, last(zs)*1e-3), title="Fluid velocity \n(z component, adimensional)", xlabel="Distance (km)")

    display(plot(p1, p2, p3, p4, layout = l, plot_title="total time=$(time_model) Ma"))

    Plots.scalefontsizes()
end




