"""Cross-fitted double machine learning for retention-offer uplift.

Sub-packages:
- ``causal``: the structural causal model generator, the declared causal specification and its
  DAG, backdoor identification, cross-fitted nuisances, the ATE and CATE estimators, the overlap
  and balance gate, the refuters, budgeted targeting and the pickle-free scoring bundle.
- ``ml``: the data contract derived from the specification, run manifests, a content-addressed
  registry, PSI and KS drift monitors and pickle-free serialization.
- ``service``: the HTTP service, with JWT roles, pseudonymous customer ids, export approvals and
  a hash-chained audit log.
- ``core``: settings, logging with PII redaction, tracing, metrics and the hardened app factory.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
