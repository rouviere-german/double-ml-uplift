# 3. Search for an adjustment set from the ancestors, not from every covariate

- Status: accepted
- Date: 2026-09-21
- Supersedes the initial implementation, which was unsound in the direction described below.

## Context

Given a DAG, a treatment and an outcome, the module has to return a set of observed variables
satisfying Pearl's backdoor criterion, or refuse. The obvious implementation — and the one this
repository shipped first — is:

1. take every admissible variable (observed, not the treatment or outcome, not downstream of
   either, not forbidden);
2. check whether conditioning on all of them blocks every backdoor path;
3. if it does, greedily drop variables that are not needed and return what is left;
4. if it does not, refuse.

Step 1 is the folk rule "adjust for everything measured before the treatment". Step 3 makes the
answer minimal. It looks complete: surely if the largest admissible set does not work, no subset
will.

That is false, and the property test
`tests/unit/test_identification.py::test_a_refusal_means_no_admissible_subset_would_have_worked`
found it by comparing the module's verdict against brute-force enumeration of every subset on
generated DAGs. The counterexample is the classic **M-bias** structure:

```text
t <- a -> m <- b -> y
```

`m` is a pre-treatment collider. It is not a descendant of the treatment or the outcome, so it is
admissible under step 1. The single backdoor trail `t <- a -> m <- b -> y` is **blocked** by the
empty set, because `m` is an unconditioned collider. Conditioning on `m` *opens* it. So:

- with `a` and `b` observed, the old search started from `{a, b, m}`, which does block (via `a`
  and `b`), then greedily dropped down to `{b}`. A valid answer, just not the empty set.
- with `a` and `b` **unobserved**, the only admissible variable is `m`. Conditioning on `{m}` opens
  the collider path, so step 2 failed and the module **refused a query it could have answered**:
  the empty set is a perfectly good adjustment set here.

A false refusal is less dangerous than a false estimate, but it is still wrong, and it is corrosive
in exactly the way ADR 0002 worries about — a tool that refuses answerable questions teaches
people to route around it.

## Decision

The search starts from the **canonical adjustment set**: the allowed variables that are also
ancestors of the treatment or the outcome.

```text
Z0 = An({T, Y}) ∩ allowed \ (De(T) ∪ {T, Y})
```

The greedy minimisation then runs on `Z0` as before.

The justification is a known result: if any subset of the allowed variables satisfies the backdoor
criterion, then `Z0` does (van der Zander, Liśkiewicz and Textor, "Constructing separators and
adjustment sets in ancestral graphs", UAI 2014; Shpitser, VanderWeele and Robins, "On the validity
of covariate adjustment", UAI 2010). The intuition for the M-bias case is direct: `m` is not an
ancestor of `t` or of `y`, so it cannot sit on an open backdoor trail as a non-collider, so it can
never be *needed* — it can only do harm. Excluding it is not a heuristic, it is what makes the
search complete.

We did not take this on trust. The completeness claim is a property test over generated DAGs that
compares the module's verdict against enumeration of all `2^|pool|` subsets, and the thirty-plus
fixture DAGs include both M-bias variants with their expected answers.

## Consequences

**Good.** Identification is now sound *and* complete relative to the declared graph and the
allowed variable set, and both directions are tested rather than argued. A refusal now genuinely
means "no admissible set would have worked", which is what the README and the threat model claim.

**Good.** The returned sets are smaller and better motivated. Pre-treatment colliders and
variables irrelevant to both treatment and outcome no longer appear in an adjustment set just
because they happened to be measured first.

**Bad.** The result is *locally* minimal, not globally smallest: greedy dropping in sorted order
returns a set from which no single variable can be removed, which need not be the smallest such
set. Finding the smallest is a harder problem and buys nothing statistically — any valid set
identifies the same estimand, and extra pre-treatment covariates that predict the outcome reduce
variance rather than increase bias. The property test asserts local minimality, which is the
honest claim.

**Bad.** Completeness is relative to the **allowed** set, which this repository deliberately keeps
narrower than the criterion requires: descendants of the outcome are excluded, and so are
pre-treatment variables whose *name* looks post-treatment. Either could in principle exclude a
variable that some valid set needed. That is a policy choice in favour of refusing over quietly
adjusting for something that looks like leakage, and it is visible — `Identification.considered`
reports what was in the pool.

**Neutral.** The change is invisible for the bundled `retention_offer` specification, whose
covariates are all genuine common causes. It matters for the graphs users write.
