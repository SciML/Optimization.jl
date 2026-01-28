using OptimizationBase
using OptimizationBase: OptimizationVerbosity
using Test
using SciMLLogging

@testset "OptimizationVerbosity Tests" begin
    @testset "Default constructor" begin
        v = OptimizationVerbosity()
        @test v isa OptimizationVerbosity

        # Spot check a few fields from each group
        @test v.convergence_failure isa SciMLLogging.WarnLevel
        @test v.unsupported_kwargs isa SciMLLogging.InfoLevel
        @test v.missing_second_order_ad isa SciMLLogging.WarnLevel
    end

    @testset "Preset constructors" begin
        # Test None - everything Silent
        v_none = OptimizationVerbosity(SciMLLogging.None())
        @test v_none.convergence_failure isa SciMLLogging.Silent
        @test v_none.unsupported_kwargs isa SciMLLogging.Silent

        # Test Minimal - only critical errors
        v_minimal = OptimizationVerbosity(SciMLLogging.Minimal())
        @test v_minimal.convergence_failure isa SciMLLogging.WarnLevel
        @test v_minimal.unsupported_kwargs isa SciMLLogging.Silent
        @test v_minimal.missing_second_order_ad isa SciMLLogging.WarnLevel

        # Test Standard - balanced verbosity (default)
        v_standard = OptimizationVerbosity(SciMLLogging.Standard())
        @test v_standard.convergence_failure isa SciMLLogging.WarnLevel
        @test v_standard.unsupported_kwargs isa SciMLLogging.InfoLevel
        @test v_standard.unsupported_bounds isa SciMLLogging.WarnLevel
    end

    @testset "Group-level settings" begin
        v = OptimizationVerbosity(convergence_numerical = SciMLLogging.ErrorLevel())
        @test v.convergence_failure isa SciMLLogging.ErrorLevel
        @test v.nan_inf_gradients isa SciMLLogging.ErrorLevel

        # Other groups should use defaults
        @test v.unsupported_kwargs isa SciMLLogging.InfoLevel
    end

    @testset "Individual field overrides" begin
        v = OptimizationVerbosity(
            convergence_numerical = SciMLLogging.Silent(),
            nan_inf_gradients = SciMLLogging.WarnLevel()  # Override
        )
        # Individual override takes precedence
        @test v.nan_inf_gradients isa SciMLLogging.WarnLevel
        # Other fields in group use group setting
        @test v.convergence_failure isa SciMLLogging.Silent
    end

    @testset "_process_verbose_param" begin
        # Test with Bool
        v_true = OptimizationBase._process_verbose_param(true)
        @test v_true isa OptimizationVerbosity

        v_false = OptimizationBase._process_verbose_param(false)
        @test v_false.convergence_failure isa SciMLLogging.Silent

        # Test with OptimizationVerbosity
        v = OptimizationVerbosity()
        @test OptimizationBase._process_verbose_param(v) === v
    end

    @testset "Validation" begin
        # Invalid group argument
        @test_throws ArgumentError OptimizationVerbosity(convergence_numerical = "invalid")

        # Invalid field name
        @test_throws ArgumentError OptimizationVerbosity(unknown_field = SciMLLogging.InfoLevel())
    end
end
