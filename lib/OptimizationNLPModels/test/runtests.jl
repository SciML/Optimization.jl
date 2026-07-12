using Pkg
using SafeTestsets

const TEST_GROUP = get(ENV, "OPTIMIZATION_TEST_GROUP", "All")

# QA (Aqua + JET) runs in an isolated environment (test/qa) so its tooling deps
# never enter the main test target's resolve. Develop the package by path so
# QA tests the current checkout without committed local source paths.
function activate_qa_env()
    Pkg.activate(joinpath(@__DIR__, "qa"))
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.develop(PackageSpec(path = joinpath(@__DIR__, "..", "..", "OptimizationBase")))
    Pkg.develop(PackageSpec(path = joinpath(@__DIR__, "..", "..", "OptimizationLBFGSB")))
    Pkg.develop(PackageSpec(path = joinpath(@__DIR__, "..", "..", "OptimizationMOI")))
    Pkg.develop(PackageSpec(path = joinpath(@__DIR__, "..", "..", "OptimizationOptimJL")))
    return Pkg.instantiate()
end

if TEST_GROUP == "Core" || TEST_GROUP == "All"
    @time @safetestset "Core" include("core_tests.jl")
end

if TEST_GROUP == "QA"
    activate_qa_env()
    @safetestset "Quality Assurance" include("qa/qa.jl")
end
