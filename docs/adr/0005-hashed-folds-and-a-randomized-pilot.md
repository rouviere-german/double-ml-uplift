# 5. Hashed fold assignment, and a randomized pilot slice as the only external check

- Status: accepted
- Date: 2026-09-21

Two decisions about *where the trust comes from*, recorded together because they answer the same
question from opposite ends: what in this system can be checked without believing the system.

## Context, part one: folds

Cross-fitting needs a partition of the rows. The default is `sklearn.model_selection.KFold` with
`shuffle=True` and a `random_state`.

That is reproducible only in a narrow sense: the same rows, in the same order, in one process.
Change the row order, take a subset, split the work across processes, or add a customer, and a
given customer moves to a different fold. Which means two runs that an analyst would describe as
"the same run" produce different numbers, and there is no cheap way to tell whether a difference
is a real change or fold churn.

It is also steerable in principle. If fold membership depends on position, whoever controls the
order of the upload controls the partition.

## Decision, part one

Fold membership is a pure function of the customer id and a configured salt:

```text
fold(unit) = int(blake2b(unit_id, key=salt, digest_size=8)) mod k
```

The digest of the whole `(unit, fold)` assignment goes into the run manifest, so two runs that
claim to be the same run can be compared without shipping the assignment.

Consequences follow directly and are tested as properties: reordering the rows permutes the
assignment identically, dropping rows leaves the survivors where they were, and adding a customer
does not move anybody. Keying the hash with a salt means an attacker who can choose customer ids
cannot aim them into a chosen fold without knowing it.

`make_fold_plan` refuses an assignment that leaves a fold empty rather than silently producing a
degenerate split, and `n_folds=1` — which disables cross-fitting entirely — exists only for the
Frisch–Waugh–Lovell differential test; `EstimationConfig` requires `n_folds >= 2` for real work.

The costs are honest ones. Fold sizes are multinomial rather than exactly equal (on 10,000 ids and
five folds the counts land within about ±3% of even, which is asserted). There is no stratification
by treatment arm, so a training fold could in principle contain a single arm; `cross_fit` raises
rather than producing a propensity model that cannot be fitted. And the salt is configuration, not
a secret — it is recorded in the configuration hash — so the anti-steering property is only as
strong as the fact that the salt is not published alongside the ids.

## Context, part two: what can be checked from outside

Everything else in this repository is an *internal* consistency check. Identification checks the
graph against itself. The overlap gate checks the data against a threshold. Placebo, random common
cause and subset stability all check the estimator against its own behaviour under perturbation —
they catch a broken pipeline, a leaky fold, an unstable estimator. None of them can see the
assumption that actually decides whether the number means anything: unconfoundedness. A confounded
estimate is perfectly self-consistent. It passes every one of those checks.

Sensitivity analysis is the usual answer. The E-value reports how strong an unmeasured confounder
would have to be to explain the estimate away — useful, but it is a statement about hypothetical
confounders, not evidence about this one, and for a small standardised effect on a continuous
outcome the answer is always "not very strong", which is why it is classified *advisory* rather
than blocking.

## Decision, part two

The simulator draws a small **randomized pilot slice** — a fraction of customers for whom the
offer was assigned by a coin flip regardless of the historical targeting rule — and two things
use it:

1. **`pilot_consistency`**, a blocking refuter: compare the observational estimate against a
   plain difference in means on the randomized slice. That difference needs no unconfoundedness
   assumption, so disagreement is evidence about the assumption itself rather than about the
   pipeline. In the Monte Carlo it detects a strong hidden confounder in 92% of draws, and its
   false-flag rate on well-specified scenarios is 0.000.
2. **Policy value.** The value of a targeting policy is estimated *only* on the pilot, as a
   difference in means among the customers the policy would target. Estimating it on the same
   observational data that trained the CATE model would let the model grade its own homework.
   At a 20% budget the pilot interval covers the true policy value in 100% of draws.

When the slice is too thin to support either — fewer than `min_pilot_per_arm` rows in the smaller
arm, or fewer than five targeted rows per arm — the refuter reports `skipped` and the value is
simply absent. Neither is estimated from a handful of rows.

## Consequences

**Good.** There is exactly one check in the system whose validity does not depend on the system
being right, and it is the check on the one assumption nothing else can see. The system's honest
claim is therefore "we detect strong unmeasured confounding 92% of the time *when a randomized
slice exists*", which is a much more defensible sentence than "the refuters passed".

**Bad, and this is the largest residual risk in the repository.** Without a randomized slice there
is no external check at all. `pilot_consistency` skips silently, the E-value is advisory, and the
remaining refuters cannot see confounding by construction. A deployment on purely observational
history is therefore strictly weaker than the evaluation suggests, and the model card says so.

**Bad.** The pilot costs money: a fraction of customers get the offer at random, including
customers the model would not have chosen and, in the heterogeneous case, customers the offer
harms. The default `pilot_fraction` in the bundled scenarios is 0.05–0.20. That is a real business
cost being paid for a statistical guarantee, and it should be an explicit decision rather than a
default someone inherits.

**Bad.** The pilot estimate is high-variance. At realistic slice sizes its interval is wide — wide
enough that "the pilot CI covers the true value" is a weak statement, and `pilot_consistency` will
only catch *large* discrepancies. It detects the hidden confounder in the bench because that
confounder biases the estimate by about +64 on a true effect of +22. Subtle confounding passes.

**Neutral.** The observational and pilot estimates share rows, so treating their difference as
having variance `se_obs² + se_pilot²` slightly overstates it. The check is therefore conservative:
it flags less often than a sharper variance estimate would, which is the safer direction for a
blocking refuter.
