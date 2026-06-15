using Pkg
using SafeTestsets

const TEST_GROUP = get(ENV, "OPTIMIZATION_TEST_GROUP", "All")

# QA (Aqua + JET) runs in an isolated environment (test/qa) so its tooling deps
# never enter the main test target's resolve. On Julia < 1.11 the [sources] table
# is ignored, so develop the package by path to test the PR branch code.
function activate_qa_env()
    Pkg.activate(joinpath(@__DIR__, "qa"))
    if VERSION < v"1.11.0-DEV.0"
        Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    end
    return Pkg.instantiate()
end

if TEST_GROUP == "Core" || TEST_GROUP == "All"
    @time @safetestset "Core" include("core_tests.jl")
end

if TEST_GROUP == "QA"
    activate_qa_env()
    @safetestset "Quality Assurance" include("qa/qa.jl")
end
