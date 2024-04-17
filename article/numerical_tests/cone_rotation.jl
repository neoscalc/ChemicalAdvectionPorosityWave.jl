
using Plots
using Interpolations
using BenchmarkTools
using Parameters
using ProgressBars
using DelimitedFiles
import Base.Threads.@threads

include("Code/Model.jl")
include("Code/UW_scheme.jl")
include("Code/SL_scheme.jl")
include("Code/WENO_scheme.jl")
include("Code/MIC_scheme.jl")

using .Model
using .UW_scheme
using .SL_scheme
using .WENO_scheme
using .MIC_scheme

function main()

    cd(@__DIR__)

    # grid
    nx = 200
    ny = 200
    Lx = 1.0
    Ly = 1.0
    Δx = Lx / nx
    Δy = Ly / ny
    x = range(Δx/2, length=nx, stop= Lx-Δx/2)
    y = range(Δy/2, length=ny, stop= Ly-Δy/2)
    grid = (x' .* ones(ny), ones(nx)' .* y)
    w = π*1e-5  # angular velocity
    vx0 = -w .* (grid[2] .- Ly/2)
    vy0 = w .* (grid[1] .- Lx/2)
    period = 1  # revolution number
    tmax = period / (w/(2*π))
    Δt = 200.0

    u0::Matrix{Float64} = zeros(ny, nx)

    R = 24*Δx
    x0 = 1/4
    y0 = 1/4

    for I in CartesianIndices(u0)
        r = sqrt((grid[1][I] - x0)^2 + (grid[2][I] - y0)^2)
        if r <= R
            u0[I] = 1
        else
            u0[I] = 0
        end
    end

    Param = Model.ModelParameters(u0=u0, Δx=Δx, Δy=Δy, Lx=Lx, Ly=Ly, vx0=vx0, vy0=vy0, tmax=tmax, Δt=Δt)

    SL = Model.SemiLagrangianScheme(nx=nx, ny=ny, u0=u0);
    WENO = Model.WENOScheme(nx=nx, ny=ny, u0=u0);
    UW = Model.UWScheme(nx=nx, ny=ny, u0=u0)
    MIC = Model.MICScheme(u0=u0,nx=nx, ny=ny, Lx=Lx, Ly=Ly)

    u_SL_QM::Matrix{Float64} = copy(u0)
    u_WENO::Matrix{Float64} = copy(u0)
    u_UW::Matrix{Float64} = copy(u0)
    u_MIC::Matrix{Float64} = copy(u0)

    # initialize marker composition
    MIC_scheme.MIC_initialize_markers!(MIC.u_mark, u_MIC, MIC, Param)

    # loop over tmax with a step of Δt
    for _ in tqdm(range(0, tmax, step=Δt))
        SL_scheme.semi_lagrangian!(u_SL_QM, SL, Param.v0, Param.v0, Param; method="quasi-monotone")
        WENO_scheme.WENO_scheme!(u_WENO, Param.v0, WENO, Param; method=:Z)
        MIC_scheme.MIC!(u_MIC, MIC,Param.v0, Param)

        l = @layout [a b; c d]

        p1 = heatmap(x, y, u_WENO, title="WENO")
        p2 = heatmap(x, y, u_SL_QM, title="SL QM")
        p3 = heatmap(x, y, u_MIC, title="MIC")
        p4 = plot(axis=([], false))

        display(plot(p1, p2, p3, p4, layout = l))
    end

    Δt = 80.0
    for _ in tqdm(range(0, tmax, step=Δt))
        UW_scheme.Upwind!(u_UW, UW.u_old, Param.v0, Δt, Param)

        display(heatmap(x, y, u_UW, title="Upwind"))
    end

    mass_UW = sum(u_UW .* Δx .* Δy) / sum(u0 .* Δx .* Δy)
    mass_WENO = sum(u_WENO .* Δx .* Δy) / sum(u0 .* Δx .* Δy)
    mass_SL_QM = sum(u_SL_QM .* Δx .* Δy) / sum(u0 .* Δx .* Δy)
    mass_MIC = sum(u_MIC .* Δx .* Δy) / sum(u0 .* Δx .* Δy)

    # writedlm("Data/C0.txt", u0)
    # writedlm("Data/UW.txt", u_UW)
    # writedlm("Data/WENO.txt", u_WENO)
    # writedlm("Data/SL_QM.txt", u_SL_QM)
    # writedlm("Data/MIC.txt", u_MIC)
    # writedlm("Data/vx.txt", vx0)
    # writedlm("Data/vy.txt", vy0)
    # writedlm("Data/gridx.txt", grid[1])
    # writedlm("Data/gridy.txt", grid[2])
    # mass = [mass_UW, mass_WENO,mass_SL_QM,mass_MIC]
    # writedlm("Data/mass.txt", mass)

    l = @layout [a b; c d]
    p1 = heatmap(x, y, u_UW, title="Upwind")
    p2 = heatmap(x, y, u_MIC, title="MIC")
    p3 = heatmap(x, y, u_WENO, title="WENO")
    p4 = heatmap(x, y, u_SL_QM, title="SL QM")

    display(plot(p1, p2, p3, p4, layout = l))

    # print mass conservation of each scheme
    println("mass conservation of UW scheme: ", mass_UW)
    println("mass conservation of WENO scheme: ", mass_WENO)
    println("mass conservation of SL quasi-monotone scheme: ", mass_SL_QM)
    println("mass conservation of MIC scheme: ", mass_MIC)

    # print maximum value of each scheme
    println("maximum value of UW scheme: ", maximum(u_UW))
    println("maximum value of WENO scheme: ", maximum(u_WENO))
    println("maximum value of SL quasi-monotone scheme: ", maximum(u_SL_QM))
    println("maximum value of MIC scheme: ", maximum(u_MIC))

    # print error of each scheme using the mean square error
    # error_UW = 1/(nx*ny) * np.sum((C0 - u_UW)**2)

    error_UW = sum((u0 .- u_UW).^2) / (nx*ny)
    error_WENO = sum((u0 .- u_WENO).^2) / (nx*ny)
    error_SL_QM = sum((u0 .- u_SL_QM).^2) / (nx*ny)
    error_MIC = sum((u0 .- u_MIC).^2) / (nx*ny)

    println("error of UW scheme: ", error_UW)
    println("error of WENO scheme: ", error_WENO)
    println("error of SL quasi-monotone scheme: ", error_SL_QM)
    println("error of MIC scheme: ", error_MIC)
end

main()
