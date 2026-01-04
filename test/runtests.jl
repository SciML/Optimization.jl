using SafeTestsets, Test, Pkg

const GROUP = get(ENV, "GROUP", "Core")

function dev_subpkg(subpkg)
    subpkg_path = joinpath(dirname(@__DIR__), "lib", subpkg)
    return Pkg.develop(PackageSpec(path = subpkg_path))
end

function activate_subpkg_env(subpkg)
    subpkg_path = joinpath(dirname(@__DIR__), "lib", subpkg)
    Pkg.activate(subpkg_path)
    Pkg.develop(PackageSpec(path = subpkg_path))
    return Pkg.instantiate()
end

@time begin
    if GROUP == "Core"
        @testset verbose = true "Optimization.jl" begin
            @safetestset "Quality Assurance" include("qa.jl")
            @safetestset "Utils Tests" begin
                include("utils.jl")
            end
            @safetestset "AD Tests" begin
                include("ADtests.jl")
            end
            @safetestset "AD Performance Regression Tests" begin
                include("AD_performance_regression.jl")
            end
            @safetestset "Optimization" begin
                include("native.jl")
            end
            @safetestset "Mini batching" begin
                include("minibatch.jl")
            end
            @safetestset "DiffEqFlux" begin
                include("diffeqfluxtests.jl")
            end
            @safetestset "Interface Compatibility" begin
                include("interface_tests.jl")
            end
        end
    elseif GROUP == "GPU"
        activate_downstream_env()
        @safetestset "DiffEqFlux GPU" begin
            include("downstream/gpu_neural_ode.jl")
        end
    else
        dev_subpkg(GROUP)
        subpkg_path = joinpath(dirname(@__DIR__), "lib", GROUP)
        Pkg.test(PackageSpec(name = GROUP, path = subpkg_path))
    end
end
