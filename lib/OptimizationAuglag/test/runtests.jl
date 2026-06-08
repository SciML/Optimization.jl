using SafeTestsets

const TEST_GROUP = get(ENV, "OPTIMIZATION_TEST_GROUP", "All")

if TEST_GROUP == "Core" || TEST_GROUP == "All"
    @time @safetestset "Core" include("core_tests.jl")
end
