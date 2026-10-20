"""Analysis and benchmark machinery for the paper — validating the estimators, not shipping them.

`recovery` (fraction-recovery benchmark), `pure_pools` (simulated known-composition pools),
`shells` (FSC / map-map correlation), `purity` (the known-fraction sweep and purity line that measure
the composition estimators), `fits` (the regression utilities those benchmarks share), and
`si_estimators` (the retired composition estimators kept for the SI bake-off against the two the
package shipped).
"""
