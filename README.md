# ChemicalAdvectionPorosityWave.jl

ChemicalAdvectionPorosityWave is the source code for the paper "" by Dominguez et al., 2023.

### Quick start

```julia
grid = Grid(nx=201, nz=201, Lx=2u"km", Lz=4000u"m", tfinal=0.62u"Myr")

rock = UnitProperties(visco_s=1e19u"Pa*s",
            ρ_s=2_700u"kg/m^3",
            shear_mod=35u"GPa",
            )

anomaly = UnitProperties(visco_s=1e19u"Pa*s",
            ρ_s=2_700u"kg/m^3",
            shear_mod=35u"GPa",
            ϕ=0.10
            )

units = Dict(:rock => rock,
             :anomaly => anomaly)

fluid = FluidProperties(visco_f=100u"Pa*s",
            ρ_f=2_200u"kg/m^3")


# initial conditions
initialize_physical_prop!(domain) do x, z
    if inellipse(;x₀=1000u"m", z₀=300u"m", rx=800u"m", rz=100u"m")(x, z)
        :anomaly
    else
        :rock
    end
end

#define boundaries for the top and bottom of the model (Mirror boundaries on the sides)
boundary_conditions = BoundaryConditions(grid=grid, fluid_flux=false, Pe_top=0, Pe_bot=0)

# run model
sol = simulate(model)
```
