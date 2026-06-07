using Pkg
using SafeTestsets, Test

const GROUP = get(ENV, "GROUP", "Core")

function activate_downstream_env()
    Pkg.activate("downstream")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    return Pkg.instantiate()
end

@time begin
    # Detect sublibrary test groups.
    # GROUP can be a bare sublibrary name (Core test group) or
    # "{sublibrary}_{TEST_GROUP}" for any custom group (e.g., QA, GPU, etc.).
    # Sublibraries declare their groups in test/test_groups.toml.
    lib_dir = joinpath(dirname(@__DIR__), "lib")

    # Check if GROUP matches a sublibrary, possibly with a _SUFFIX for the test group.
    # Scan underscores right-to-left to find the longest matching sublibrary prefix.
    function _detect_sublibrary_group(group, lib_dir)
        isdir(joinpath(lib_dir, group)) && return (group, "Core")
        for i in length(group):-1:1
            if group[i] == '_' && isdir(joinpath(lib_dir, group[1:(i - 1)]))
                return (group[1:(i - 1)], group[(i + 1):end])
            end
        end
        return (group, "Core")
    end
    base_group, test_group = _detect_sublibrary_group(GROUP, lib_dir)

    if isdir(joinpath(lib_dir, base_group))
        Pkg.activate(joinpath(lib_dir, base_group))
        # On Julia < 1.11, the [sources] section in Project.toml is not supported.
        # Manually Pkg.develop local path dependencies so CI tests the PR branch code.
        # We resolve transitively: each developed dependency's own [sources] are also
        # developed, so sibling sublibraries reachable through a chain of [sources]
        # (e.g. OptimizationBase under OptimizationNLPModels) are correctly found.
        if VERSION < v"1.11.0-DEV.0"
            developed = Set{String}()
            # Never develop the active project itself; Pkg refuses with "package <X>
            # has the same name or UUID as the active project".
            push!(developed, normpath(joinpath(lib_dir, base_group)))
            specs = Pkg.PackageSpec[]
            queue = [joinpath(lib_dir, base_group)]
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
    elseif GROUP == "Core"
        @testset verbose = true "Optimization.jl" begin
            @safetestset "Quality Assurance" include("qa.jl")
            @safetestset "Utils Tests" include("utils.jl")
            @safetestset "Verbosity Tests" include("verbosity.jl")
            @safetestset "AD Tests" include("ADtests.jl")
            @safetestset "AD Performance Regression Tests" include("AD_performance_regression.jl")
            @safetestset "Optimization" include("native.jl")
            @safetestset "Mini batching" include("minibatch.jl")
            # DiffEqFlux test temporarily skipped due to ForwardDiff gradient dispatch issue
            # with Float32 ComponentArrays. See GitHub issue for tracking.
            # @safetestset "DiffEqFlux" include("diffeqfluxtests.jl")
            @safetestset "Interface Compatibility" include("interface_tests.jl")
            @safetestset "Sense Handling" include("sense_tests.jl")
        end
    elseif GROUP == "GPU"
        activate_downstream_env()
        @safetestset "DiffEqFlux GPU" include("downstream/gpu_neural_ode.jl")
    end
end
