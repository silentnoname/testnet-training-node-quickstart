from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from scripts.train_basic_vla import index_hf_dataset, load_samples_from_hf
from scripts.train_frozen_qwen_vla import (
    row_episode_id,
    row_horizon,
    row_step,
)


class FakeColumnDataset:
    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0]) if rows else []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return self.rows[key]

    def __iter__(self):
        return iter(self.rows)

    def select(self, indices):
        return FakeColumnDataset([self.rows[index] for index in indices])


class DatasetSchemaTests(unittest.TestCase):
    def test_basic_loader_indexes_current_hf_column_names(self):
        dataset = FakeColumnDataset(
            [
                {"episode_index": 0, "step_index": 0, "done": False},
                {"episode_index": 0, "step_index": 1, "done": True},
                {"episode_index": 1, "step_index": 0, "done": False},
                {"episode_index": 1, "step_index": 1, "done": False},
                {"episode_index": 1, "step_index": 2, "done": True},
            ]
        )

        index = index_hf_dataset(dataset)

        self.assertEqual(index.episode_column, "episode_index")
        self.assertEqual(index.step_column, "step_index")
        self.assertEqual(index.episode_rows, {"0": [0, 1], "1": [2, 3, 4]})
        self.assertEqual(index.episode_horizons, {"0": 2, "1": 3})

    def test_basic_loader_can_derive_episodes_from_done(self):
        dataset = FakeColumnDataset(
            [
                {"step_index": 0, "done": False},
                {"step_index": 1, "done": True},
                {"step_index": 0, "done": False},
                {"step_index": 1, "done": True},
            ]
        )

        index = index_hf_dataset(dataset)

        self.assertIsNone(index.episode_column)
        self.assertEqual(index.episode_rows, {"0": [0, 1], "1": [2, 3]})

    def test_basic_loader_reads_current_hf_schema_end_to_end(self):
        rows = []
        for episode_index, length in ((0, 2), (1, 3)):
            for step_index in range(length):
                rows.append(
                    {
                        "episode_index": episode_index,
                        "step_index": step_index,
                        "task": "lift_cube",
                        "difficulty": "low",
                        "instruction": "Lift the cube.",
                        "image": np.zeros((4, 4, 3), dtype=np.uint8),
                        "action": np.full(7, step_index, dtype=np.float32),
                        "proprio": np.zeros(25, dtype=np.float32),
                        "done": step_index == length - 1,
                    }
                )
        dataset = FakeColumnDataset(rows)
        fake_datasets = SimpleNamespace(load_dataset=lambda *args, **kwargs: dataset)

        with mock.patch.dict("sys.modules", {"datasets": fake_datasets}):
            samples = load_samples_from_hf(
                "fake/current-schema",
                step_stride=1,
                max_samples=0,
            )

        self.assertEqual(samples.images.shape, (5, 4, 4, 3))
        self.assertEqual(samples.proprio.shape, (5, 32))
        self.assertEqual(samples.actions.shape, (5, 7))
        self.assertEqual(samples.manifest_summary["trajectory_count"], 2)
        self.assertEqual(
            samples.manifest_summary["episode_column"],
            "episode_index",
        )

    def test_qwen_loader_accepts_current_and_legacy_aliases(self):
        current = {"episode_index": 12, "step_index": 34}
        legacy = {"episode_id": "episode-a", "step": 5, "horizon": 180}

        self.assertEqual(row_episode_id(current), "12")
        self.assertEqual(row_step(current), 34)
        self.assertEqual(row_horizon(current), 200)
        self.assertEqual(row_episode_id(legacy), "episode-a")
        self.assertEqual(row_step(legacy), 5)
        self.assertEqual(row_horizon(legacy), 180)


if __name__ == "__main__":
    unittest.main()
