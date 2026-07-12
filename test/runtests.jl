using Pkg
using SafeTestsets, Test
using SciMLTesting

const GROUP = current_group(; default = "Core")
const LIB_DIR = joinpath(dirname(@__DIR__), "lib")

@time begin
    # Monorepo sublibrary routing: a bare sublibrary name (its Core group) or
    # "<sublibrary>_<group>" selects a lib/<Sublibrary> sub-package, which is
    # activated and `Pkg.test`ed with OPTIMIZATION_TEST_GROUP set. This pre-step is
    # kept explicit (not delegated to run_tests' lib_dir routing) to preserve the
    # `--check-bounds=auto` julia arg, `force_latest_compatible_version = false`, and
    # the pre-1.11 transitive [sources] develop walk byte-for-byte.
    base_group, test_group = detect_sublibrary_group(GROUP, LIB_DIR)

    if !isempty(base_group) && isdir(joinpath(LIB_DIR, base_group))
        Pkg.activate(joinpath(LIB_DIR, base_group))
        # On Julia < 1.11 the [sources] table is ignored, so manually develop local
        # path dependencies (transitively, following each dep's own [sources]) so CI
        # tests the PR-branch code.
        if VERSION < v"1.11.0-DEV.0"
            developed = Set{String}()
            push!(developed, normpath(joinpath(LIB_DIR, base_group)))
            specs = Pkg.PackageSpec[]
            queue = [joinpath(LIB_DIR, base_group)]
            while !isempty(queue)
                pkg_dir = popfirst!(queue)
                toml_path = joinpath(pkg_dir, "Project.toml")
                isfile(toml_path) || continue
                toml = Pkg.TOML.parsefile(toml_path)
                if haskey(toml, "sources")
                    for (dep_name, source_spec) in toml["sources"]
                        if source_spec isa Dict && haskey(source_spec, "path")
                            dep_path = normpath(joinpath(pkg_dir, source_spec["path"]))
                            if isdir(dep_path) && !(dep_path in developed)
                                push!(developed, dep_path)
                                @info "Queuing local source dependency" dep_name dep_path
                                push!(specs, Pkg.PackageSpec(path = dep_path))
                                push!(queue, dep_path)
                            end
                        end
                    end
                end
            end
            isempty(specs) || Pkg.develop(specs)
        end
        withenv("OPTIMIZATION_TEST_GROUP" => test_group) do
            Pkg.test(
                base_group,
                julia_args = ["--check-bounds=auto"],
                force_latest_compatible_version = false,
                allow_reresolve = true
            )
        end
    else
        run_tests(;
            default = "Core",
            core = function ()
                @safetestset "Utils Tests" include("utils.jl")
                @safetestset "Verbosity Tests" include("verbosity.jl")
                @safetestset "Optimization" include("native.jl")
                @safetestset "Mini batching" include("minibatch.jl")
                # DiffEqFlux test temporarily skipped due to ForwardDiff gradient dispatch
                # issue with Float32 ComponentArrays. See GitHub issue for tracking.
                # @safetestset "DiffEqFlux" include("diffeqfluxtests.jl")
                @safetestset "Interface Compatibility" include("interface_tests.jl")
                @safetestset "Sense Handling" include("sense_tests.jl")
            end,
            groups = Dict(
                # The AD tests are their own group with their own environment
                # (test/AD/Project.toml) so their Enzyme-family dependencies stay out
                # of the main test env, and so the group can be excluded from the Julia
                # `pre` channel in test_groups.toml: Enzyme is expected to fail on
                # prereleases until a fixed Enzyme release lands.
                # See https://github.com/SciML/Optimization.jl/issues/1260.
                "AD" => (;
                    env = joinpath(@__DIR__, "AD"),
                    body = function ()
                        @safetestset "AD Tests" include("AD/ADtests.jl")
                        @safetestset "AD Performance Regression Tests" include("AD/AD_performance_regression.jl")
                    end,
                ),
                "GPU" => (;
                    env = joinpath(@__DIR__, "downstream"),
                    body = function ()
                        @safetestset "DiffEqFlux GPU" include("downstream/gpu_neural_ode.jl")
                    end,
                ),
            ),
            qa = (;
                env = joinpath(@__DIR__, "qa"),
                body = joinpath(@__DIR__, "qa", "qa.jl"),
            ),
            all = String[],
        )
    end
end
