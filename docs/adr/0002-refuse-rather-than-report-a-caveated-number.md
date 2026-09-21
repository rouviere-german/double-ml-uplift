# 2. Refuse rather than report a caveated number

- Status: accepted
- Date: 2026-09-21

## Context

Every causal estimation tool has to decide what to do when its assumptions are not supported by
the data. There are three options, and the whole character of the tool follows from the choice:

1. **Report the number with a warning.** This is what almost every library does. `dowhy`,
   `EconML` and a hand-rolled statsmodels pipeline will all happily return an ATE when overlap has
   failed, and leave the diagnostics as a separate call the analyst may or may not make.
2. **Report the number with the diagnostics attached**, so a caller who inspects them can decide.
3. **Refuse:** raise, name the reasons, and do not compute a point estimate at all.

The pressure towards (1) is real. A refusal is an annoying answer. Someone has a campaign to
launch, the pipeline returns an exception instead of a number, and the path of least resistance is
to route around the tool — copy the estimator call into a notebook, drop the gate, ship the number.

But the failure mode of (1) is worse than annoyance. An estimate computed under a positivity
violation is mostly extrapolation, and it comes with a *narrow* interval, because the standard
error reflects sampling variability and not the fact that there are no comparable untreated
customers. In the Monte Carlo, the `weak_overlap` scenario gives AIPW a bias of +6.5 and an
interval that covers the truth 100% of the time only because it is enormous; IPW's RMSE is 12.
Nothing in those numbers says "do not trust me". A warning in a log line does not travel with the
number into a slide deck.

Option (2) sounds like the reasonable middle. In practice it is option (1): a number and a
diagnostic in the same response means the number gets used and the diagnostic gets skimmed. The
asymmetry is what matters — the cost of a skipped diagnostic is a wrong campaign, the cost of a
refusal is an inconvenience.

## Decision

The engine **fails closed**. Identification failure, data-contract failure, and overlap or
balance failure each raise a typed exception carrying the reasons, and no point estimate is
computed at all.

Concretely:

- The order of operations in `UpliftEngine.fit_prepared` puts every gate *before* the estimators.
  The gate runs after cross-fitting (it needs the propensities) but before `estimate_all`, so
  there is never a point estimate in memory that could be logged, cached or returned after a
  refusal.
- `Identification.require()` raises `NotIdentifiedError` with the open backdoor paths named.
- `GateDecision.require()` raises `NotEstimableError` with one reason per failing threshold.
- The HTTP layer maps these to `422` with the reasons and no estimate fields, and counts them in
  `uplift_refusals_total{reason}` so refusals are observable rather than invisible.
- At scoring time the same bounds produce a per-customer **abstention** instead of a number.

One escape hatch exists: `EstimationConfig.enforce_gate=False`. It is what lets
`bench/bench_montecarlo.py` measure the bias and coverage the gate protects against — without it
the "what does the gate buy" table in the README could not be produced. The service never exposes
it, and `tests/integration/test_engine.py::test_the_gate_can_be_measured_without_being_enforced`
pins the behaviour: the result is still marked not estimable and still carries its reasons.

## Consequences

**Good.** A number that comes out of this system has passed identification, the contract, overlap
and balance. That is a property of the *system*, not of the diligence of whoever called it. The
refusal names what failed, so it is actionable: "positivity: 20.3% of estimated propensities fall
outside [0.05, 0.95]" tells an analyst that the historical targeting rule was too deterministic
and that the answer is an experiment, not a different estimator.

**Bad.** Thresholds are now load-bearing, and they are judgement calls. `max_outside_fraction =
0.05` and `max_abs_smd = 0.20` are conventional (Austin 2009 for the SMD) but not derived. A
campaign just inside them is still partly extrapolating and gets no refusal; a campaign just
outside gets no number at all. `GatePolicy` is configurable per run precisely because the right
values are contextual, which means a caller who dislikes a refusal can loosen the gate — but they
have to do it explicitly, it changes the configuration hash recorded with the estimate, and it is
therefore visible in review.

**Bad.** Refusals push work onto the caller at the least convenient moment. We accept that. The
alternative is a tool that is easy to use and occasionally, silently, confidently wrong about
which customers to contact.

**Neutral.** The diagnostics are computed and stored either way, so a refused run still produces
the full overlap report. Nothing is thrown away except the point estimate that should not exist.
