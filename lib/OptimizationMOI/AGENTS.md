Import changes must preserve runtime evaluation of registered symbolic coefficient functions, including calls that appear only in generated expressions. Static ExplicitImports checks do not cover this path; keep analytic coefficient and cache preservation regressions in the official Core group.

Run validation from the repository root:

```sh
OPTIMIZATION_TEST_GROUP=Core julia --project=lib/OptimizationMOI -e 'using Pkg; Pkg.test()'
OPTIMIZATION_TEST_GROUP=QA julia --project=lib/OptimizationMOI -e 'using Pkg; Pkg.test()'
```
