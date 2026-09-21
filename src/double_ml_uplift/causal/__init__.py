"""Causal estimation kit: SCM generator, identification, cross-fitted estimators, gates.

The pieces are deliberately separate so each one can be tested against ground truth:

- ``scm``: a seeded structural causal model that keeps both potential outcomes for scoring only.
- ``spec``: the declared causal specification (treatment, outcome, DAG, forbidden columns).
- ``graph``: d-separation and backdoor reachability on a directed acyclic graph.
- ``identification``: backdoor adjustment sets, refusals and the post-treatment leakage guard.
- ``folds``: hashed, reproducible cross-fitting fold assignment.
- ``nuisance``: the small typed surface over scikit-learn and LightGBM learners.
- ``estimators``: naive, outcome regression, clipped IPW, AIPW, partialling-out DML, DR-learner.
- ``diagnostics``: overlap and balance gates that return "not estimable" with reasons.
- ``refutation``: placebo, random common cause, subset stability, pilot consistency, E-values.
- ``targeting``: budgeted CATE targeting with pilot-based value estimation and a fairness check.
- ``engine``: the pipeline that wires all of the above together and fails closed.
"""
