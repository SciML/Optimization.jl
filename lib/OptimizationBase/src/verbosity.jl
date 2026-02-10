@verbosity_specifier OptimizationVerbosity begin
    toggles = (
        # Parameter compatibility warnings
        :unsupported_kwargs,

        # Convergence and numerical issues
        :convergence_failure, :nan_inf_gradients, :singularity_at_bounds, :unrecognized_stop_reason,

        # Constraints and bounds
        :unsupported_bounds, :equality_constraints_ignored, :inequality_constraints_ignored,

        # Automatic differentiation
        :missing_second_order_ad, :incompatible_ad_backend,

        # Feature support
        :unsupported_callbacks,

        # Solver-specific verbosity
        :ipopt_verbosity,
    )

    presets = (
        None = (
            unsupported_kwargs = Silent(),
            convergence_failure = Silent(),
            nan_inf_gradients = Silent(),
            singularity_at_bounds = Silent(),
            unrecognized_stop_reason = Silent(),
            unsupported_bounds = Silent(),
            equality_constraints_ignored = Silent(),
            inequality_constraints_ignored = Silent(),
            missing_second_order_ad = Silent(),
            incompatible_ad_backend = Silent(),
            unsupported_callbacks = Silent(),
            ipopt_verbosity = CustomLevel(0),
        ),
        Minimal = (
            unsupported_kwargs = Silent(),
            convergence_failure = WarnLevel(),
            nan_inf_gradients = WarnLevel(),
            singularity_at_bounds = WarnLevel(),
            unrecognized_stop_reason = WarnLevel(),
            unsupported_bounds = Silent(),
            equality_constraints_ignored = Silent(),
            inequality_constraints_ignored = Silent(),
            missing_second_order_ad = WarnLevel(),
            incompatible_ad_backend = WarnLevel(),
            unsupported_callbacks = Silent(),
            ipopt_verbosity = CustomLevel(0),
        ),
        Standard = (
            unsupported_kwargs = InfoLevel(),
            convergence_failure = WarnLevel(),
            nan_inf_gradients = WarnLevel(),
            singularity_at_bounds = WarnLevel(),
            unrecognized_stop_reason = WarnLevel(),
            unsupported_bounds = WarnLevel(),
            equality_constraints_ignored = WarnLevel(),
            inequality_constraints_ignored = WarnLevel(),
            missing_second_order_ad = WarnLevel(),
            incompatible_ad_backend = WarnLevel(),
            unsupported_callbacks = WarnLevel(),
            ipopt_verbosity = CustomLevel(5),
        ),
        Detailed = (
            unsupported_kwargs = InfoLevel(),
            convergence_failure = WarnLevel(),
            nan_inf_gradients = WarnLevel(),
            singularity_at_bounds = WarnLevel(),
            unrecognized_stop_reason = WarnLevel(),
            unsupported_bounds = WarnLevel(),
            equality_constraints_ignored = WarnLevel(),
            inequality_constraints_ignored = WarnLevel(),
            missing_second_order_ad = WarnLevel(),
            incompatible_ad_backend = WarnLevel(),
            unsupported_callbacks = WarnLevel(),
            ipopt_verbosity = CustomLevel(7),
        ),
        All = (
            unsupported_kwargs = InfoLevel(),
            convergence_failure = WarnLevel(),
            nan_inf_gradients = WarnLevel(),
            singularity_at_bounds = WarnLevel(),
            unrecognized_stop_reason = WarnLevel(),
            unsupported_bounds = WarnLevel(),
            equality_constraints_ignored = WarnLevel(),
            inequality_constraints_ignored = WarnLevel(),
            missing_second_order_ad = WarnLevel(),
            incompatible_ad_backend = WarnLevel(),
            unsupported_callbacks = WarnLevel(),
            ipopt_verbosity = CustomLevel(12),
        ),
    )

    groups = (
        convergence_numerical = (
            :convergence_failure, :nan_inf_gradients, :singularity_at_bounds, :unrecognized_stop_reason,
        ),
        constraints_bounds = (
            :unsupported_bounds, :equality_constraints_ignored, :inequality_constraints_ignored,
        ),
        automatic_differentiation = (
            :missing_second_order_ad, :incompatible_ad_backend,
        ),
        feature_support = (
            :unsupported_callbacks, :unsupported_kwargs,
        ),
        solver_verbosity = (
            :ipopt_verbosity,
        ),
    )
end


"""
    OptimizationVerbosity <: AbstractVerbositySpecifier

Verbosity configuration for Optimization.jl solvers, providing fine-grained control over
diagnostic messages and warnings during optimization.

# Fields

## Convergence and Numerical Issues Group
- `convergence_failure`: Messages when algorithm fails to converge
- `nan_inf_gradients`: Messages when NaN or Inf values appear in gradients
- `singularity_at_bounds`: Messages when function has singularities at bounds
- `unrecognized_stop_reason`: Messages when stop reason is not recognized

## Constraints and Bounds Group
- `unsupported_bounds`: Messages when bounds are not supported by the algorithm
- `equality_constraints_ignored`: Messages when equality constraints are not passed to the algorithm
- `inequality_constraints_ignored`: Messages when inequality constraints are not passed to the algorithm

## Automatic Differentiation Group
- `missing_second_order_ad`: Messages when second-order AD is required but not provided
- `incompatible_ad_backend`: Messages when AD backend is incompatible with algorithm requirements

## Feature Support Group
- `unsupported_callbacks`: Messages when callbacks are not supported by the algorithm
- `unsupported_kwargs`: Messages when common optimization parameters (abstol, reltol, maxtime, maxiters) are not supported by the algorithm

## Solver Verbosity Group
- `ipopt_verbosity`: Controls Ipopt solver output verbosity (0=silent, 5=default, 12=maximum). Use a SciMLLogging.CustomLevel to specify an integer verbosity level. 

# Constructors

    OptimizationVerbosity(preset::AbstractVerbosityPreset)

Create an `OptimizationVerbosity` using a preset configuration:
- `SciMLLogging.None()`: All messages disabled
- `SciMLLogging.Minimal()`: Only critical convergence issues and AD warnings
- `SciMLLogging.Standard()`: Balanced verbosity (default)
- `SciMLLogging.Detailed()`: Comprehensive information
- `SciMLLogging.All()`: Maximum verbosity

    OptimizationVerbosity(; preset=nothing, convergence_numerical=nothing, constraints_bounds=nothing, automatic_differentiation=nothing, feature_support=nothing, kwargs...)

Create an `OptimizationVerbosity` with group level or individual toggle level control.

# Examples

```julia
# Use a preset
verbose = OptimizationVerbosity(SciMLLogging.Standard())

# Set entire groups
verbose = OptimizationVerbosity(
    convergence_numerical = SciMLLogging.WarnLevel(),
    feature_support = SciMLLogging.InfoLevel()
)

# Set individual fields
verbose = OptimizationVerbosity(
    convergence_failure = SciMLLogging.ErrorLevel(),
    unsupported_kwargs = SciMLLogging.Silent()
)

# Mix group and individual settings
verbose = OptimizationVerbosity(
    feature_support = SciMLLogging.InfoLevel(),  # Set all feature warnings to InfoLevel
    unsupported_callbacks = SciMLLogging.Silent()  # Override specific field
)
```
"""
function OptimizationVerbosity end

const DEFAULT_VERBOSE = OptimizationVerbosity()

@inline function _process_verbose_param(verbose::SciMLLogging.AbstractVerbosityPreset)
    return OptimizationVerbosity(verbose)
end

@inline function _process_verbose_param(verbose::Bool)
    return verbose ? DEFAULT_VERBOSE : OptimizationVerbosity(SciMLLogging.None())
end

@inline _process_verbose_param(verbose::OptimizationVerbosity) = verbose
