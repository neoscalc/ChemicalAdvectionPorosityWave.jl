module ChemicalAdvectionPorosityWave

using Reexport: @reexport
@reexport using Plots
@reexport using Unitful
@reexport using Parameters
@reexport using Symbolics
@reexport using OrdinaryDiffEq, DiffEqCallbacks, LinearSolve
@reexport using BenchmarkTools

# Write your package code here.
include("solvers/two_phase_flow.jl")
include("input/initial_conditions.jl")
include("simulate/simulate.jl")
include("callbacks/visualization/plotting.jl")
include("callbacks/advection/velocity.jl")
include("callbacks/advection/advection.jl")
include("callbacks/advection/UW_scheme.jl")
include("callbacks/advection/SL_scheme.jl")
include("callbacks/advection/WENO_scheme.jl")
include("callbacks/advection/MIC_scheme.jl")
include("callbacks/advection/stepsize_limiter.jl")
include("callbacks/output/output.jl")
include("utils.jl")

function __init__()
    # initialise global logger for OrdinaryDiffEq
    global_logger(TerminalLogger())
end

export Grid, Domain
export Model
export simulate
export porosity_wave
export plotting_tpf, composition_fluid
export advection
# export UWScheme, SemiLagrangianScheme, WENOScheme, advection
export velocity_call_func, advection_call_func, dtmaxC
export hdf5_start, hdf5_initial_conditions, save_data

end
