# trials_legacy_metrics.csv

The (degree x training-set-size) sweep as it stood before the metric set was reworked:
500 trials (5 degrees x 2 sizes x 50 trials) x 3 solvers, 1000 pooled contexts each,
seeds unchanged (`sweep.seed_for`, RNG_SEED = 143). Kept because it is the only copy of
these numbers; regenerating it means rerunning the array.

## Read the `regret_Y` column with care

It does NOT hold what any current version of the code calls `regret_Y`. Writing
z*(c) = argmin_w <c, w> and w_hat = z*(f(X)) for a solver's decision, the column is

    regret_Y  =  100 * sum(<f*, w_hat> - <f*, z*(Y)>) / sum(<Y, z*(Y)>)

i.e. both decisions scored under f*. Today's `regret_Y` is the SPO regret,
<Y, w_hat> - <Y, z*(Y)>, which is non-negative by construction -- 990 of the 1500 rows
here are negative, which is how you can tell the two apart at a glance.

It is not today's `regret_Y_lowvar` either: that is <f*, w_hat> - <Y, z*(Y)>, which keeps
the benchmark noisy and so is an unbiased, lower-variance estimate of `regret_Y`. The
column here subtracts <f*, z*(Y)> instead, a quantity current runs store as
`sum_fstar_zY`.

`loss_Y` and `regret_fstar` are unchanged and compare directly against a current sweep.

## What a current sweep writes instead

`trials.csv`, with four metric columns (`loss_Y`, `regret_Y`, `regret_Y_lowvar`,
`regret_fstar`) plus the five raw cost sums they are ratios of (`sum_Y_zmethod`,
`sum_fstar_zmethod`, `sum_Y_zY`, `sum_fstar_zY`, `sum_fstar_zfstar`) and `num_contexts`,
so any normalization -- including none -- is arithmetic on the CSV rather than a rerun.
