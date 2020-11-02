using SafeTestsets

println("Rosenbrock Tests")
@safetestset "Rosenbrock" begin include("rosenbrock.jl") end
println("AD Tests")
@safetestset "AD Tests" begin include("ADtests.jl") end
println("Mini batching Tests")
@safetestset "Mini batching" begin include("minibatch.jl") end