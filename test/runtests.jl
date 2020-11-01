using SafeTestsets

@safetestset "Rosenbrock" begin include("rosenbrock.jl") end
@safetestset "AD Tests" begin include("ADtests.jl") end
@safetestset "Mini batching" begin include("minibatch.jl") end