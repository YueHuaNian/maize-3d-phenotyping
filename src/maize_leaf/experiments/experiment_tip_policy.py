"""复现叶尖强制截短消融；仅显式运行时启用，结果必须放入新目录。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from . import validate_holdout
import maize_leaf.surface as surface

# 保留原拟合器，工作进程只在初始化时安装消融版本。
_original = surface.fit_dornbusch_blade


def reject_boundary_tip(s_obs, w_obs, length_obs, prior=None, allow_tip_completion=True):
    """将触及长度上界的叶尖改为不补全，复现已撤回策略。"""
    blade = _original(s_obs, w_obs, length_obs, prior=prior, allow_tip_completion=allow_tip_completion)
    if blade.completion_status == "boundary_limited":
        blade = _original(s_obs, w_obs, length_obs, prior=prior, allow_tip_completion=False)
        blade.completion_status = "unresolved_at_length_bound"
    return blade


def install_worker_policy():
    """在独立工作进程内安装叶尖消融，避免影响主进程的正常重建。"""
    surface.fit_dornbusch_blade = reject_boundary_tip


def main(argv=None):
    """校验独立输出路径，记录消融来源并启动冻结划分留出验证。"""
    argv = list(sys.argv[1:] if argv is None else argv)
    # 帮助请求直接交给完整留出解析器，不能创建目录或安装策略。
    if "--help" in argv or "-h" in argv:
        return validate_holdout.main(argv)
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args(argv)
    # 创建输出前先校验全部参数，未知参数不能留下半成品目录。
    validate_holdout.build_parser().parse_args(argv)
    if args.output.exists():
        parser.error("消融输出必须使用尚不存在的新目录")
    args.output.mkdir(parents=True)
    (args.output / "policy_override.json").write_text(json.dumps({
        "policy": "reject boundary-limited tip instead of flagging its constrained estimate",
        "driver": "src/maize_leaf/experiments/experiment_tip_policy.py",
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2), encoding="utf-8")
    return validate_holdout.main(argv, worker_initializer=install_worker_policy)
