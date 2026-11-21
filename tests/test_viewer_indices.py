from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from src.viewer_indices import read_index_results
from view_phenotypes import _pick_leaf, read_results
from view_phenotypes_v2 import DEFAULT_INDICES, DEFAULT_SOURCE, load_view_indices


class ViewerIndicesTests(unittest.TestCase):
    def test_default_lookup_uses_plant_and_leaf_ids(self):
        _, leaves, _ = read_results(DEFAULT_SOURCE, "plant")
        plants, metrics = load_view_indices(DEFAULT_SOURCE, leaves)
        self.assertEqual(len(plants), 14)
        self.assertEqual(len(metrics), 90)
        self.assertAlmostEqual(float(plants["2-1-2"]["pai"]), 28.13, places=2)
        self.assertAlmostEqual(float(metrics["2-1-2", 1]["lmi"]), 26.22, places=2)

    def test_reject_indices_from_different_input_and_allow_legacy_viewer(self):
        with TemporaryDirectory() as temp:
            source = Path(temp)
            self.assertEqual(load_view_indices(source, {}), ({}, {}))
            (source / "input_manifest.csv").write_text("different input", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"different input_manifest\.csv"):
                read_index_results(DEFAULT_INDICES, source, {})

    def test_click_uses_visible_label_or_projected_leaf_point(self):
        figure = Figure()
        FigureCanvasAgg(figure)
        ax = figure.add_subplot(projection="3d")
        ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1))
        low = ax.scatter([0.1], [0.1], [0.15])
        high = ax.scatter([0.8], [0.8], [0.85])
        label = ax.text(0.8, 0.8, 0.7, "8", bbox={"boxstyle": "round"})
        figure.canvas.draw()
        targets = {2: (low, None), 8: (high, label)}

        box = label.get_bbox_patch().get_window_extent()
        x, y = box.x0 + box.width / 2, box.y0 + box.height / 2
        self.assertEqual(_pick_leaf(targets, x, y, 5), 8)
        high.set_visible(False)
        label.set_visible(False)
        self.assertIsNone(_pick_leaf(targets, x, y, 5))
        x, y = low.get_offset_transform().transform(low.get_offsets())[0]
        self.assertEqual(_pick_leaf(targets, x, y, 5), 2)
