"""Reproduce the rejected boundary-tip truncation ablation in a fresh directory.

This driver intentionally overrides the normal fitting policy. The default
pipeline keeps the constrained estimate and labels it boundary_limited.
"""
import hashlib
import json
from pathlib import Path
import sys
from validate_holdout import main
import maize_leaf.surface as surface

_original = surface.fit_dornbusch_blade


def reject_boundary_tip(s_obs, w_obs, length_obs, prior=None, allow_tip_completion=True):
    blade = _original(s_obs, w_obs, length_obs, prior=prior, allow_tip_completion=allow_tip_completion)
    if blade.completion_status == "boundary_limited":
        blade = _original(s_obs, w_obs, length_obs, prior=prior, allow_tip_completion=False)
        blade.completion_status = "unresolved_at_length_bound"
    return blade


# Spawned workers execute the driver too, so the same ablation is installed
# before each worker calls validate_holdout.fit_one_leaf.
surface.fit_dornbusch_blade = reject_boundary_tip


if __name__ == "__main__":
    if "--output" not in sys.argv:
        raise SystemExit("An explicit fresh --output directory is required for the ablation")
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    if output.exists():
        raise SystemExit("Choose a new directory; ablation outputs cannot share the normal cache")
    output.mkdir(parents=True)
    (output / "policy_override.json").write_text(json.dumps({
        "policy": "reject boundary-limited tip instead of flagging its constrained estimate",
        "driver": "scripts/experiment_tip_policy.py",
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2), encoding="utf-8")
    main()
