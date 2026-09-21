"""Command line entry point: ``double-ml-uplift`` or ``python -m double_ml_uplift``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from double_ml_uplift.service.auth import Role

__all__ = ["main"]


def _serve(host: str | None, port: int | None) -> int:
    import uvicorn  # noqa: PLC0415 - only needed when serving

    from double_ml_uplift.causal.spec import bundled_spec  # noqa: PLC0415
    from double_ml_uplift.core.config import get_settings  # noqa: PLC0415
    from double_ml_uplift.core.logging import configure_logging  # noqa: PLC0415
    from double_ml_uplift.core.tracing import install_global_tracer_provider  # noqa: PLC0415
    from double_ml_uplift.service.app import create_service_app  # noqa: PLC0415

    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)
    app = create_service_app(bundled_spec("retention_offer"), settings=settings)
    install_global_tracer_provider(app.state.tracer_provider)
    uvicorn.run(
        app,
        host=host or settings.host,
        port=port or settings.port,
        log_config=None,
        access_log=False,
        server_header=False,
    )
    return 0


def _demo(workdir: Path) -> int:
    from double_ml_uplift.causal.demo import run_uplift_demo  # noqa: PLC0415
    from double_ml_uplift.core.logging import configure_logging  # noqa: PLC0415

    configure_logging("WARNING", json_output=True, stream=sys.stderr)
    print(json.dumps({"demo": "uplift", **run_uplift_demo(workdir)}, indent=2, sort_keys=True))
    return 0


def _spec(name: str) -> int:
    from double_ml_uplift.causal.identification import identify_backdoor  # noqa: PLC0415
    from double_ml_uplift.causal.spec import SpecError, bundled_spec  # noqa: PLC0415

    try:
        spec = bundled_spec(name)
    except SpecError as exc:
        print(json.dumps({"spec": name, "valid": False, "error": str(exc)}, indent=2))
        return 1
    identification = identify_backdoor(spec)
    print(
        json.dumps(
            {
                "spec": name,
                "valid": True,
                "sha256": spec.sha256(),
                "treatment": spec.treatment,
                "outcome": spec.outcome,
                "identified": identification.identified,
                "adjustment_set": list(identification.adjustment_set),
                "reasons": list(identification.reasons),
                "backdoor_paths": identification.backdoor_paths,
            },
            indent=2,
        )
    )
    return 0 if identification.identified else 1


def _audit_verify(path: Path, expect_head: str | None) -> int:
    from double_ml_uplift.service.audit import AuditError, verify_file  # noqa: PLC0415

    try:
        count = verify_file(path, expect_head=expect_head)
    except AuditError as exc:
        print(json.dumps({"audit": str(path), "valid": False, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps({"audit": str(path), "valid": True, "entries": count}, indent=2))
    return 0


def _issue_token(subject: str, role: str, lifetime: float) -> int:
    from double_ml_uplift.core.config import get_settings  # noqa: PLC0415
    from double_ml_uplift.service.auth import issue_token  # noqa: PLC0415

    settings = get_settings()
    try:
        token = issue_token(settings, subject, cast("Role", role), lifetime_seconds=lifetime)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps({"subject": subject, "role": role, "token": token}, indent=2))
    print(
        "Local development only. Production tokens come from the identity provider.",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="double-ml-uplift")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP service")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    sub.add_parser("generate-api-key", help="print a new API key and its SHA-256 digest")

    token = sub.add_parser("issue-token", help="mint a local JWT (development only)")
    token.add_argument("subject")
    token.add_argument("--role", choices=["analyst", "operator"], default="analyst")
    token.add_argument("--lifetime", type=float, default=600.0)

    spec = sub.add_parser("spec", help="validate a bundled causal specification and identify it")
    spec.add_argument("name", default="retention_offer", nargs="?")

    audit = sub.add_parser("audit", help="audit log tools")
    audit_sub = audit.add_subparsers(dest="audit_command", required=True)
    verify = audit_sub.add_parser("verify", help="verify a hash-chained audit log")
    verify.add_argument("path", type=Path)
    verify.add_argument("--expect-head", default=None)

    demo = sub.add_parser("demo", help="run the offline demo")
    demo.add_argument("--workdir", type=Path, default=Path("var/demo"))

    args = parser.parse_args(argv)
    if args.command == "serve":
        return _serve(args.host, args.port)
    if args.command == "generate-api-key":
        from double_ml_uplift.core.security import generate_api_key  # noqa: PLC0415

        raw, digest = generate_api_key()
        print(json.dumps({"api_key": raw, "api_key_sha256": digest}, indent=2))
        print("Store only api_key_sha256 in APP_API_KEY_SHA256.", file=sys.stderr)
        return 0
    if args.command == "issue-token":
        return _issue_token(args.subject, args.role, args.lifetime)
    if args.command == "spec":
        return _spec(args.name)
    if args.command == "audit":
        return _audit_verify(args.path, args.expect_head)
    return _demo(args.workdir)


if __name__ == "__main__":
    raise SystemExit(main())
