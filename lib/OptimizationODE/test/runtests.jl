using Pkg
using SafeTestsets
using SciMLTesting

const TEST_GROUP = get(ENV, "OPTIMIZATION_TEST_GROUP", "All")

# QA (Aqua + JET) runs in an isolated environment (test/qa). activate_group_env
# develops the package under test (via `parent`) plus its in-repo `[sources]` siblings
# by path — native `[sources]` on Julia >= 1.11, the develop_sources! backport on 1.10.
function activate_qa_env()
    return activate_group_env(joinpath(@__DIR__, "qa"))
end

if TEST_GROUP == "Core" || TEST_GROUP == "All"
    @time @safetestset "Core" include("core_tests.jl")
end

if TEST_GROUP == "QA"
    activate_qa_env()
    @safetestset "Quality Assurance" include("qa/qa.jl")
end
