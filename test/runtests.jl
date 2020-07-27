using GalacticOptim
using Test

@testset "GalacticOptim.jl" begin
    include("rosenbrock.jl")
    include("ADtests.jl")
end
