"""研究验证与诊断绘图实现；通过 scripts/ 下的两个统一入口调用。"""
import argparse
from importlib import import_module
import sys

# 按需加载所选实验，查看总帮助不会导入绘图库或启动计算。
VALIDATION_TASKS = {
    "fit": ("validate_fit", "重建前后逐叶贴合审计"),
    "holdout": ("validate_holdout", "冻结划分遮挡留出验证"),
    "compare": ("compare_holdout", "两轮留出结果配对比较"),
    "clustering": ("validate_euclidean_clustering", "欧氏聚类参数标定"),
    "sor": ("validate_sor_sensitivity", "分断片 SOR 参数标定"),
    "trait-windows": ("calibrate_trait_windows", "表型窗口标定及配套诊断图"),
    "tip-ablation": ("experiment_tip_policy", "历史叶尖截短策略消融（新目录）"),
}
PLOT_TASKS = {
    "holdout": ("plot_holdout", "留出误差与案例诊断"),
    "clustering": ("plot_euclidean_clustering", "欧氏聚类敏感性图"),
    "sor": ("plot_sor_sensitivity", "SOR 敏感性图"),
    "support": ("plot_support_comparison", "观测支撑域前后对照"),
    "index": ("plot_ideotype_index", "理想株型结构符合度指数四联诊断图"),
}


def dispatch(tasks, argv=None):
    """解析任务名并原样转交后续参数，保留各实验自己的参数校验。"""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="选择研究任务；在任务名后加 --help 查看参数。",
        epilog="；".join(f"{name}：{entry[1]}" for name, entry in tasks.items()),
    )
    parser.add_argument("task", choices=tasks, help="研究任务名称")
    # 只解析首项，确保子任务的 --help 与参数不会被总入口截获。
    args = parser.parse_args(argv[:1])
    module = import_module("." + tasks[args.task][0], __name__)
    return module.main(argv[1:])
