using SciMLTesting

run_tests(;
    env = "OPTIMIZATION_TEST_GROUP",
    core = joinpath(@__DIR__, "core_tests.jl"),
    groups = Dict(
        "AD" => (;
            env = joinpath(@__DIR__, "AD"),
            body = joinpath(@__DIR__, "AD", "tests.jl"),
        ),
    ),
    qa = (; env = joinpath(@__DIR__, "qa"), body = joinpath(@__DIR__, "qa", "qa.jl")),
    all = ["Core"],
)
