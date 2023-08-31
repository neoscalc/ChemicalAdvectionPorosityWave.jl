"""
    DensityX(oxd_wt::Vector{_T}, T::_T, P::_T) where {_T<:Number}

Calculate silicate melt density `rho` after Iacovino & Till (2019)

Input: 
- `oxd_wt`  Melt composition as 9-element vector containing concentrations 
            in [wt%] of the following oxides ordered in the exact sequence 
            [SiO2 TiO2 Al2O3 FeO MgO CaO Na2O K2O H2O]
              1    2    3     4   5   6   7    8   9  
- `T`       Melt temperature in [degC]
- `P`       Pressure in [kbar]

Output:
- `rho`     Melt density in [kg/m3]

Reference:
- Iacovino K & Till C (2019). DensityX: A program for calculating the densities of magmatic liquids up to 1,627 °C and 30 kbar. Volcanica 2(1), p 1-10. [doi:10.30909/vol.02.01.0110](https://dx.doi.org/10.30909/vol.02.01.0110)

Modified from Tobias Keller: [kellertobs @ GitHub](https://github.com/kellertobs), 03/2023
Original Python code by K Iacovino: [github.com/kaylai/DensityX](https://github.com/kaylai/DensityX), 03/2023

Example: calculate melt density of hydrous N-MORB at 1200 C and 1.5 kbar
===
```julia
julia> oxd_wt = [50.42, 1.53, 15.13, 9.81, 7.76, 11.35, 2.83, 0.140, 1.0]  # [wt%]
julia> T = 1200.0  # [deg C]
julia> P = 1.5     # [kbar]
julia> rho = DensityX(oxd_wt,T,P)  # [kg/m3]
```
"""
function melt_density_i(oxd_wt::Vector{_T}, oxd_mol::Vector{_T}) where {_T<:Number}

    # Set parameter values

    # Molecular Weights [g/mol]
    MW  = (60.0855, 79.88, 101.96, 71.85, 40.3, 56.08, 61.98, 94.2)

    # Partial Molar Volumes
    # Volumes for SiO2, Al2O3, MgO, CaO, Na2O, K2O at Tref = 1773 K (Lange, 1997; CMP)
    # Volume for FeO at Tref = 1723 K (Guo et al., 2014)
    # Volume for TiO2 at Tref = 1773 K (Lange and Carmichael, 1987)
    MV  = (26.86, 28.32, 37.42, 12.68, 12.02, 16.90, 29.65, 47.28)

    # dV/dT values
    # MgO, CaO, Na2O, K2O Table 4 (Lange, 1997)
    # SiO2, TiO2, Al2O3 Table 9 (Lange and Carmichael, 1987)
    # FeO from Guo et al (2014)
    dVdT  = (0.0, 0.00724, 0.00262, 0.00369, 0.00327, 0.00374, 0.00768, 0.01208)

    # dV/dP values
    # Anhydrous component data from Kess and Carmichael (1991)
    dVdP  = (-0.000189, -0.000231, -0.000226, -0.000045, 0.000027, 0.000034, -0.00024, -0.000675)

    # Tref values
    Tref  = (1773.15, 1773.15, 1773.15, 1723.15, 1773.15, 1773.15, 1773.15, 1773.15)

    # Convert temperature to [K]
    T_K     = 1200.0 .+ 273.15

    # Convert pressure to [bar]
    P_bar   = 6.0 .* 1e3

    # Calculate melt density

    # Divide normalized wt% values by molecular weights
    oxd_mol   .= oxd_wt ./ MW
    oxd_mol .= oxd_mol ./ sum(oxd_mol)

    # Calculate the partial Vliq for each oxide
    part_Vliq = (MV .+ (dVdT .* (T_K .- Tref)) .+ (dVdP .* (P_bar .- 1.0))) .* oxd_mol

    # Calculate partial X*MW
    part_XMW  = oxd_mol .* MW

    # Calculate melt density in [kg/m3]
    rho 	  = sum(part_XMW) ./ sum(part_Vliq) * 1000.0
    return rho

end

