using SafeTestsets

const TEST_GROUP = get(ENV, "OPTIMIZATION_TEST_GROUP", "ALL")

if TEST_GROUP == "Core" || TEST_GROUP == "ALL"
    @time @safetestset "Core" include("core_tests.jl")
end
