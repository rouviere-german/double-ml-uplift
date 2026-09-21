"""HTTP service: role-separated access to estimates, scores and targeting exports.

- ``auth``: HS256 JWT verification and the analyst/operator role split.
- ``identifiers``: HMAC pseudonymisation of customer ids, so raw ids never leave the process.
- ``audit``: a hash-chained append-only log of every export request, decision and release.
- ``approvals``: export approvals bound to the exact payload, to a second principal, single use.
- ``metrics``: the Prometheus families this service adds to the baseline HTTP metrics.
- ``models``: request and response schemas (closed, size-capped).
- ``app``: the routes and the in-process state they share.
"""
