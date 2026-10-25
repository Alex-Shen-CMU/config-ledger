"""``config-ledger`` command line: look up runs, inspect and resolve runs under old versions.

    config-ledger lookup       --ledger <root> --namespace <ns> CONFIG [--parent DIR] [--json]
    config-ledger lookup       --config-dir D --config-name N [OVERRIDES...] [-m] [--setup M[:F]]
    config-ledger versions     --ledger <root> [selection]
    config-ledger mark-stale   --ledger <root> [selection] [--apply]
    config-ledger upgrade      --ledger <root> [selection] [--include-stale] [--apply]
    config-ledger purge-stale  --ledger <root> [selection] [--apply]
    config-ledger rebuild      --ledger <root> --root <outputs> [--root ...] [--namespace NS ...]
                               [--apply]
    config-ledger dump-configs --setup MODULE:FUNC [--setup ...] --out DIR [--check]

Everything but ``versions`` and ``lookup`` is a dry run unless ``--apply`` is given. A target
is a full ``_target_`` or a class name; a version is a full chain (``Trainer:1|Gold:2``) or the leaf
class's integer version. Selection: ``[--namespace NS ...] [--target T ...] [--version V ...]``.
See :mod:`config_ledger.maintenance`.

``rebuild`` regenerates sidecars and version index entries from the ``run_manifest.json`` files
under the output roots, after losing the ledger or moving it or the outputs. It never
overwrites: conflicts are reported and exit 1, as do manifests it can't rebuild from.

``dump-configs`` (extra ``[hydra]``) writes every ``ConfigStore`` entry that the ``--setup``
functions register to static YAML under ``--out``; ``--check`` writes nothing and exits 1 if
any file is missing or out of date. See :mod:`config_ledger.hydra.dump`.

``lookup`` is read-only: it prints the folder id owning a config (``--parent``: the output
folder) and exits 0, or exits 1 if no folder owns it (for a sweep: 1 if any grid point is
missing), 2 on errors. CONFIG is a composed ``.json`` or ``.yaml`` (extra ``[omegaconf]``);
``--config-name`` composes with Hydra (extra ``[hydra]``) and reads ``--ledger`` /
``--namespace`` from the config's ``ledger:`` block unless given. The current directory and
every ``--pythonpath`` are importable, so ``_target_`` classes of the project resolve. See
:mod:`config_ledger.lookup` and :mod:`config_ledger.hydra.lookup`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence

from config_ledger import maintenance
from config_ledger.lookup import LookupResult, load_config_file, lookup
from config_ledger.store import Ledger


def _selection(args: argparse.Namespace) -> maintenance.Selection:
    return maintenance.Selection(namespaces=args.namespace or (), targets=args.target or (),
                                 versions=args.version or ())


def _print_outcomes(outcomes: list[maintenance.Outcome], apply: bool,
                    dry_run_statuses: set[str]) -> None:
    for item in outcomes:
        print(f"{item.status:13s} {item.namespace}/{item.folder_id}  {item.output_folder}")
        for cause in item.causes:
            print(f"{'':16s}because: {cause}")
        for line in filter(None, item.detail.splitlines()):
            print(f"{'':16s}{line}")
    counts = maintenance.summarize(outcomes)
    summary = ", ".join(f"{n} {status}" for status, n in sorted(counts.items())) or "nothing to do"
    print(f"\n{summary}")
    if not apply and dry_run_statuses & set(counts):
        print("Dry run. Re-run with --apply to make these changes.")


def _versions(args: argparse.Namespace) -> int:
    rows = maintenance.report(Ledger(args.ledger), _selection(args))
    if not rows:
        print("no versioned runs recorded")
        return 0
    print(f"{'namespace':12s} {'state':12s} {'fresh':>6s} {'stale':>6s} {'migr.':>6s}  "
          "target @ version")
    for row in rows:
        print(f"{row.namespace:12s} {row.state:12s} {row.fresh:6d} {row.stale:6d} "
              f"{row.migrated:6d}  {row.target} @ {row.version_chain}")
    blocking = [r for r in rows if r.state != "live" and r.fresh]
    if blocking:
        print(f"\n{len(blocking)} old version(s) still have fresh runs and block new runs of "
              "their classes: `config-ledger upgrade` or `config-ledger mark-stale` them.")
    return 0


def _mark_stale(args: argparse.Namespace) -> int:
    outcomes = maintenance.mark_stale(Ledger(args.ledger), _selection(args), apply=args.apply)
    _print_outcomes(outcomes, args.apply, {maintenance.WOULD_MARK})
    return 1 if any(o.status == maintenance.REFUSED for o in outcomes) else 0


def _upgrade(args: argparse.Namespace) -> int:
    outcomes = maintenance.upgrade(Ledger(args.ledger), _selection(args), apply=args.apply,
                                   include_stale=args.include_stale)
    _print_outcomes(outcomes, args.apply, {maintenance.WOULD_UPGRADE})
    return 0


def _purge_stale(args: argparse.Namespace) -> int:
    outcomes = maintenance.purge_stale(Ledger(args.ledger), _selection(args), apply=args.apply)
    _print_outcomes(outcomes, args.apply, {maintenance.WOULD_PURGE})
    return 0


def _rebuild(args: argparse.Namespace) -> int:
    outcomes = maintenance.rebuild(Ledger(args.ledger), args.root, namespaces=args.namespace or (),
                                   apply=args.apply)
    _print_outcomes([o for o in outcomes if o.status != maintenance.CURRENT], args.apply,
                    {maintenance.WOULD_CREATE, maintenance.WOULD_ADD, maintenance.WOULD_MOVE})
    current = sum(o.status == maintenance.CURRENT for o in outcomes)
    print(f"Scanned {', '.join(map(str, args.root))}: {current} sidecar(s) / index record(s) "
          "already current")
    failed = {maintenance.CONFLICT, maintenance.FAILED}
    return 1 if any(o.status in failed for o in outcomes) else 0


def _lookup_points(args: argparse.Namespace) -> list[tuple[list[str], LookupResult]]:
    _extend_sys_path(args.pythonpath)
    if args.config_name is None:
        if args.multirun or args.setup or args.config_dir:
            raise ValueError("-m, --setup and --config-dir need --config-name")
        if len(args.args) != 1 or args.ledger is None or args.namespace is None:
            raise ValueError("lookup takes --ledger, --namespace and one CONFIG file "
                             "(or --config-name to compose with Hydra)")
        config = load_config_file(args.args[0])
        return [([], lookup(Ledger(args.ledger), args.namespace, config))]
    try:
        from config_ledger.hydra.lookup import lookup_hydra, run_setup
    except ImportError as exc:
        raise ImportError("--config-name needs Hydra: pip install 'config-ledger[hydra]'") from exc
    run_setup(args.setup or ())
    points = lookup_hydra(args.config_dir or os.getcwd(), args.config_name, args.args,
                          multirun=args.multirun, ledger_root=args.ledger,
                          namespace=args.namespace)
    return [(point.overrides, point.result) for point in points]


def _lookup(args: argparse.Namespace) -> int:
    try:
        points = _lookup_points(args)
    except Exception as exc:  # a CLI reports every failure the same way: message, exit 2
        print(f"config-ledger lookup: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    sweep = args.multirun
    warnings = dict.fromkeys(w for _, result in points for w in result.warnings)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if args.json:
        payloads = [{**({"overrides": overrides} if sweep else {}), **result.to_json(args.parent)}
                    for overrides, result in points]
        print(json.dumps(payloads if sweep else payloads[0], indent=2))
    elif sweep:
        for overrides, result in points:
            where = result.output_folder(args.parent) or (
                result.entry.folder_id if result.entry else "MISSING")
            print(f"{str(where):36s}  {' '.join(overrides)}")
    else:
        ((_, result),) = points
        if result.entry is not None:
            print(result.output_folder(args.parent) or result.entry.folder_id)
    missing = sum(not result.found for _, result in points)
    if missing:
        what = f"{missing} of {len(points)} grid point(s)" if sweep else "this config"
        print(f"not found: no folder owns {what} in namespace {points[0][1].namespace!r}"
              + ("" if sweep else f" (hash {points[0][1].hash})"), file=sys.stderr)
    return 1 if missing else 0


def _dump_configs(args: argparse.Namespace) -> int:
    try:
        _extend_sys_path(args.pythonpath)
        try:
            from config_ledger.hydra.dump import UNCHANGED, capture_registrations, dump_config_store
            from config_ledger.hydra.lookup import run_setup
        except ImportError as exc:
            raise ImportError("dump-configs needs Hydra: pip install 'config-ledger[hydra]'") \
                from exc
        entries = capture_registrations(lambda: run_setup(args.setup))
        if not entries:
            raise ValueError(f"--setup {' '.join(args.setup)} registered no ConfigStore entries")
        dumped = dump_config_store(entries, args.out, check=args.check)
    except Exception as exc:  # a CLI reports every failure the same way: message, exit 2
        print(f"config-ledger dump-configs: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    changed = [d for d in dumped if d.status != UNCHANGED]
    for d in changed:
        print(f"{'would be ' if args.check else ''}{d.status}: {d.path}")
    print(f"{len(dumped) - len(changed)} of {len(dumped)} file(s) already current")
    return 1 if args.check and changed else 0


def _extend_sys_path(extra: Sequence[str] | None) -> None:
    for path in reversed([os.getcwd(), *(extra or ())]):
        if path not in sys.path:
            sys.path.insert(0, path)


def _add_selection(p: argparse.ArgumentParser, apply: bool = True) -> None:
    p.add_argument("--ledger", required=True, help="ledger root directory")
    p.add_argument("--namespace", action="append", help="namespace (repeatable; default: all)")
    p.add_argument("--target", action="append",
                   help="config class: _target_ or class name (repeatable)")
    p.add_argument("--version", action="append",
                   help="version chain or leaf version number (repeatable)")
    if apply:
        p.add_argument("--apply", action="store_true", help="make the changes (default: dry run)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="config-ledger", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("lookup", help="print the folder that owns a config (never allocates)")
    p.add_argument("args", nargs="*", metavar="CONFIG|OVERRIDE",
                   help="a composed .json/.yaml config, or Hydra overrides with --config-name")
    p.add_argument("--ledger", help="ledger root (Hydra: default from the `ledger:` block)")
    p.add_argument("--namespace", help="namespace (Hydra: default from the `ledger:` block)")
    p.add_argument("--parent", help="print <PARENT>/<folder_id> instead of the folder id")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--pythonpath", action="append",
                   help="extra import path for _target_ classes (repeatable; cwd is always on)")
    p.add_argument("--config-dir", help="Hydra config directory (default: cwd)")
    p.add_argument("--config-name", help="Hydra primary config: compose instead of reading a file")
    p.add_argument("-m", "--multirun", action="store_true",
                   help="look up every grid point of the sweep (with --config-name)")
    p.add_argument("--setup", action="append", metavar="MODULE[:FUNC]",
                   help="import MODULE (and call FUNC) first, e.g. to register ConfigStore "
                        "nodes (repeatable)")
    p.set_defaults(func=_lookup)

    p = sub.add_parser("versions", help="report runs per (config class, version) and status")
    _add_selection(p, apply=False)
    p.set_defaults(func=_versions)

    p = sub.add_parser("mark-stale", help="mark fresh configs under old versions as stale")
    _add_selection(p)
    p.set_defaults(func=_mark_stale)

    p = sub.add_parser("upgrade", help="upgrade fresh runs under old versions via upgrade_config")
    _add_selection(p)
    p.add_argument("--include-stale", action="store_true",
                   help="also upgrade runs that have configs marked stale (default: skip them)")
    p.set_defaults(func=_upgrade)

    p = sub.add_parser("purge-stale",
                       help="delete outputs, sidecars and index entries of stale runs")
    _add_selection(p)
    p.set_defaults(func=_purge_stale)

    p = sub.add_parser("rebuild",
                       help="regenerate sidecars and the version index from run manifests")
    p.add_argument("--ledger", required=True, help="ledger root directory")
    p.add_argument("--root", action="append", required=True,
                   help="output root to scan for run_manifest.json (repeatable)")
    p.add_argument("--namespace", action="append", help="namespace (repeatable; default: all)")
    p.add_argument("--apply", action="store_true", help="make the changes (default: dry run)")
    p.set_defaults(func=_rebuild)

    p = sub.add_parser("dump-configs",
                       help="write ConfigStore entries a setup function registers to static YAML")
    p.add_argument("--setup", action="append", required=True, metavar="MODULE[:FUNC]",
                   help="import MODULE (and call FUNC); every entry it registers is dumped "
                        "(repeatable)")
    p.add_argument("--out", required=True, help="config directory to write <group>/<name>.yaml to")
    p.add_argument("--check", action="store_true",
                   help="write nothing; exit 1 if any file is missing or out of date")
    p.add_argument("--pythonpath", action="append",
                   help="extra import path (repeatable; cwd is always on)")
    p.set_defaults(func=_dump_configs)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
