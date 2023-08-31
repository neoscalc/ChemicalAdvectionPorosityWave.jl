using HDF5
using Parameters

#! consider viscosity and density of fluid constant
function hdf5_start(Grid, Domain, path_hdf5)

    h5open(path_hdf5, "w") do file
        g = create_group(file, "Two-phase_Flow") # create a group

        attributes(g)["LengthX"] = Grid.Lx
        attributes(g)["HeightZ"] = Grid.Lz
        attributes(g)["Dx"] = Grid.Δx
        attributes(g)["Dz"] =Grid.Δz
        attributes(g)["Nx"] = Grid.nx
        attributes(g)["Nz"] = Grid.nz
        attributes(g)["TotalTimeYears"] = Grid.tfinal
        attributes(g)["ShearViscosityFluid"] = maximum(Domain.visco_f)
        attributes(g)["BackGroundPermeability"] = Domain.kc * Domain.ϕ0^Domain.n
        attributes(g)["BackGroundPorosity"] = Domain.ϕ0
        attributes(g)["DensityFluid"] = maximum(Domain.ρ_f)
        attributes(g)["ExponentPermeability"] = Domain.n
        attributes(g)["ExponentElasticity"] = Domain.b
    end

end

function hdf5_initial_conditions(Grid, Domain, Model)

    @unpack path_data = Model

    h5open(path_data, "r+") do file
        t0 = create_group(file, "Two-phase_Flow/t$(lpad("0", 4, "0"))") # create a group
        attributes(t0)["Time"] = 0
        create_group(t0, "EffectivePressure")
        create_group(t0, "FluidFlux")
        create_group(t0, "Porosity")
        create_group(t0, "Permeability")
        create_group(t0, "ChemicalAdvection")
        create_group(t0, "BulkViscosity")
        create_group(t0, "DensityRock")
        create_group(t0, "ViscosityFluid")

        # describe type of data
        attributes(t0["Porosity"])["DataType"] = "Scalar"
        attributes(t0["EffectivePressure"])["DataType"] = "Scalar"
        attributes(t0["FluidFlux"])["DataType"] = "Vector"
        attributes(t0["ChemicalAdvection"])["DataType"] = "Scalar"
        attributes(t0["Permeability"])["DataType"] = "Scalar"
        attributes(t0["BulkViscosity"])["DataType"] = "Scalar"
        attributes(t0["DensityRock"])["DataType"] = "Scalar"
        attributes(t0["ViscosityFluid"])["DataType"] = "Scalar"

        attributes(t0["Porosity"])["Center"] = "Node"
        attributes(t0["EffectivePressure"])["Center"] = "Node"
        attributes(t0["FluidFlux"])["Center"] = "Node"
        attributes(t0["ChemicalAdvection"])["Center"] = "Node"
        attributes(t0["Permeability"])["Center"] = "Node"
        attributes(t0["BulkViscosity"])["Center"] = "Node"
        attributes(t0["DensityRock"])["Center"] = "Node"
        t0["Porosity"]["Poro"] = column_to_row(Model.ϕ_ini)
        t0["EffectivePressure"]["Pe"] = column_to_row(Model.Pe_ini)
        t0["FluidFlux"]["Z"] = vec(column_to_row(zeros((Grid.nz+1, Grid.nx+1))))
        t0["FluidFlux"]["X"] = vec(column_to_row(zeros((Grid.nz+1, Grid.nx+1))))
        t0["FluidFlux"]["Y"] = vec(column_to_row(zeros((Grid.nz+1, Grid.nx+1))))
        t0["ChemicalAdvection"]["compo_f"] = column_to_row(Domain.compo_f)
        t0["Permeability"]["Permeability"] = column_to_row(Domain.kc .* Model.ϕ_ini.^Domain.n)
        t0["ViscosityFluid"]["ViscosityFluid"] = column_to_row(Domain.visco_f)
    end

end

function hdf5_timestep(Grid, Domain, Parameters, dt, tcurrent)

    @unpack c0 = Parameters

    h5open(Parameters.path_data, "r+") do file
        data_timestep = create_group(file, "Two-phase_Flow/t$(lpad("$(Int(Parameters.t_count[1]))", 4, "0"))") # create a group
        attributes(data_timestep)["TimeYears"] = tcurrent
        attributes(data_timestep)["DtYears"] = dt
        create_group(data_timestep, "EffectivePressure")
        create_group(data_timestep, "FluidFlux")
        create_group(data_timestep, "ChemicalAdvection")
        create_group(data_timestep, "Porosity")
        create_group(data_timestep, "Permeability")
        create_group(data_timestep, "BulkViscosity")
        create_group(data_timestep, "DensityRock")
        create_group(data_timestep, "ViscosityFluid")

        # describe type of data
        attributes(data_timestep["Porosity"])["DataType"] = "Scalar"
        attributes(data_timestep["EffectivePressure"])["DataType"] = "Scalar"
        attributes(data_timestep["FluidFlux"])["DataType"] = "Vector"
        attributes(data_timestep["ChemicalAdvection"])["DataType"] = "Scalar"
        attributes(data_timestep["Permeability"])["DataType"] = "Scalar"
        attributes(data_timestep["BulkViscosity"])["DataType"] = "Scalar"
        attributes(data_timestep["DensityRock"])["DataType"] = "Scalar"
        attributes(data_timestep["ViscosityFluid"])["DataType"] = "Scalar"

        attributes(data_timestep["Porosity"])["Center"] = "Node"
        attributes(data_timestep["EffectivePressure"])["Center"] = "Node"
        attributes(data_timestep["FluidFlux"])["Center"] = "Node"
        attributes(data_timestep["ChemicalAdvection"])["Center"] = "Node"
        attributes(data_timestep["Permeability"])["Center"] = "Node"
        attributes(data_timestep["BulkViscosity"])["Center"] = "Node"
        attributes(data_timestep["DensityRock"])["Center"] = "Node"
        data_timestep["Porosity"]["Poro"] = column_to_row(Parameters.ϕ)
        data_timestep["EffectivePressure"]["Pe"] = column_to_row(Parameters.Pe)
        data_timestep["ChemicalAdvection"]["compo_f"] = column_to_row(Domain.compo_f)
        data_timestep["FluidFlux"]["Z"] = vec(column_to_row(Parameters.q_f[:z] .* c0))
        data_timestep["FluidFlux"]["X"] = vec(column_to_row(Parameters.q_f[:x] .* c0))
        data_timestep["FluidFlux"]["Y"] = vec(column_to_row(zeros((Grid.nz, Grid.nx))))
        data_timestep["Permeability"]["Permeability"] = column_to_row(Domain.kc .* Parameters.ϕ.^Domain.n)
        data_timestep["ViscosityFluid"]["ViscosityFluid"] = column_to_row(Domain.visco_f)
    end
end


function save_data(integrator)
    @unpack grid, parameters, domain = integrator.p
    @unpack t_count, compaction_t, compaction_ϕ, compaction_Pe, Pe, ϕ = parameters

    Pe_ad = @view integrator.u[:, :, 1]
    ϕ_ad = @view integrator.u[:, :, 2]

    ϕ .= compaction_ϕ .* ϕ_ad
    Pe .= compaction_Pe .* Pe_ad

    t_count[1] += 1  # count time
    dt_years = integrator.dt * compaction_t / (3600 * 24 * 365.25)
    tcurrent = integrator.t * compaction_t / (3600 * 24 * 365.25)  # count time in years

    hdf5_timestep(grid,  domain,  parameters, dt_years, tcurrent)
end

