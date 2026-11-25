"""验证研究任务分发及 Windows 子进程中的消融策略隔离。"""
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr
import io
import importlib
import multiprocessing
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from maize_leaf.experiments import VALIDATION_TASKS, dispatch
from maize_leaf.experiments import experiment_tip_policy
import maize_leaf.surface as surface


def worker_policy_module():
    """返回子进程当前拟合器来源，验证策略是否安装到正确进程。"""
    return surface.fit_dornbusch_blade.__module__


class ExperimentEntryTests(unittest.TestCase):
    """覆盖统一入口转参、工作函数导入和独立消融策略。"""

    def test_dispatch_preserves_task_help_and_arguments(self):
        """检查任务名后的帮助和参数完整转交给所选模块。"""
        main = Mock(return_value=2)
        with patch("maize_leaf.experiments.import_module", return_value=SimpleNamespace(main=main)) as loader:
            result = dispatch(VALIDATION_TASKS, ["holdout", "--help", "--workers", "2"])
        loader.assert_called_once_with(".validate_holdout", "maize_leaf.experiments")
        main.assert_called_once_with(["--help", "--workers", "2"])
        self.assertEqual(result, 2)

    def test_moved_worker_functions_can_be_pickled(self):
        """检查迁入包后的工作函数仍可按模块名序列化，适配 Windows spawn。"""
        for module_name, names in {
            "validate_holdout": ("fit_one_leaf",),
            "validate_euclidean_clustering": ("_eval_radius_group", "_eval_ref_one", "_eval_l3a_one", "_eval_l3b_one"),
            "validate_sor_sensitivity": ("_eval_ref_one", "_eval_l3a_one", "_eval_l3b_one"),
            "calibrate_trait_windows": ("_eval_synth_one", "_eval_real_plant"),
        }.items():
            module = importlib.import_module("maize_leaf.experiments." + module_name)
            for name in names:
                function = getattr(module, name)
                self.assertIs(pickle.loads(pickle.dumps(function)), function)

    def test_tip_ablation_is_confined_to_spawned_worker(self):
        """检查显式消融只修改子进程，不污染导入它的主进程。"""
        original = surface.fit_dornbusch_blade
        self.assertIs(original, experiment_tip_policy._original)
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=experiment_tip_policy.install_worker_policy) as pool:
            self.assertEqual(pool.submit(worker_policy_module).result(timeout=30),
                             "maize_leaf.experiments.experiment_tip_policy")
        self.assertIs(surface.fit_dornbusch_blade, original)

    def test_tip_ablation_rejects_bad_arguments_before_creating_output(self):
        """检查错误参数不会创建消融输出目录或改变默认拟合器。"""
        with patch.object(experiment_tip_policy.Path, "mkdir") as mkdir, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                experiment_tip_policy.main(["--output", "unused_ablation", "--unknown-option"])
        self.assertEqual(result.exception.code, 2)
        mkdir.assert_not_called()
        self.assertIs(surface.fit_dornbusch_blade, experiment_tip_policy._original)
