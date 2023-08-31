using ChemicalAdvectionPorosityWave
using Test

@testset "input" begin
    grid = Grid(nx=201, nz=201, Lx=1u"km", Lz=1000u"m", tfinal=2u"Myr")

    @test grid.nx == 201
    @test grid.Δx == 5.0
    @test grid.x == 0.0:5.0:1000.0
    @test grid.tfinal == 6.31152e13

end
