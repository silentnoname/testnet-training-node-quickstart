from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file

from scripts.qwen_frozen_policy_adapter import (
    ActionHead,
    FrozenQwenVLAPolicy,
    ImageOnlyQwenProcessor,
    auxiliary_vector,
    last_token_pool,
    qwen_last_hidden_state,
)
from scripts.train_frozen_qwen_vla import (
    cache_spec,
    episode_split_masks,
    main,
    parse_args,
    stable_episode_hash,
    validate_cache_manifest,
)


class FakeProcessor:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        del messages, tokenize, add_generation_prompt
        return "<image> control"

    def __call__(self, text, images, padding, return_tensors):
        batch_size = len(text)
        del images, padding, return_tensors
        return {
            "input_ids": torch.tensor([[1, 2, 3]] * batch_size, dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 3), dtype=torch.long),
            "pixel_values": torch.zeros((batch_size, 3), dtype=torch.float32),
        }


class FakeMultimodalBase(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.visual = nn.Identity()
        self.projection = nn.Linear(1, hidden_size, bias=False)
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask, pixel_values, **kwargs):
        self.forward_calls += 1
        del attention_mask, pixel_values, kwargs
        values = input_ids.float().unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=self.projection(values))


class FakeQwen(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.model = FakeMultimodalBase(hidden_size)
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=hidden_size)
        )


class LegacyVisual(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.register_parameter(
            "anchor", nn.Parameter(torch.zeros((), dtype=torch.float32))
        )

    def forward(self, pixel_values, grid_thw):
        del pixel_values
        token_count = int(grid_thw.prod().item() // 4)
        return torch.ones((token_count, self.hidden_size))


class LegacyTextBase(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(128, hidden_size)

    def forward(self, inputs_embeds, **kwargs):
        del kwargs
        return SimpleNamespace(last_hidden_state=inputs_embeds)


class LegacyQwen(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.model = LegacyTextBase(hidden_size)
        self.visual = LegacyVisual(hidden_size)
        self.config = SimpleNamespace(image_token_id=99)

    def get_rope_index(self, input_ids, **kwargs):
        del kwargs
        return torch.zeros((3, *input_ids.shape), dtype=torch.long), None


class FrozenQwenUnitTests(unittest.TestCase):
    def test_image_only_processor_expands_visual_placeholders(self):
        class ImageProcessor:
            merge_size = 2

            def __call__(self, images, return_tensors):
                del return_tensors
                return {
                    "pixel_values": torch.zeros((len(images), 3)),
                    "image_grid_thw": torch.tensor([[1, 4, 4]] * len(images)),
                }

        class Tokenizer:
            image_token = "<|image_pad|>"

            def __init__(self):
                self.seen_text = None

            def apply_chat_template(self, *args, **kwargs):
                del args, kwargs
                return ""

            def __call__(self, text, padding, return_tensors):
                del padding, return_tensors
                self.seen_text = text
                return {
                    "input_ids": torch.zeros((len(text), 1), dtype=torch.long),
                    "attention_mask": torch.ones((len(text), 1), dtype=torch.long),
                }

        tokenizer = Tokenizer()
        processor = ImageOnlyQwenProcessor(ImageProcessor(), tokenizer)
        result = processor(
            images=[np.zeros((4, 4, 3), dtype=np.uint8)],
            text=["a <|image_pad|> b"],
            padding=True,
            return_tensors="pt",
        )
        self.assertEqual(tokenizer.seen_text[0].count("<|image_pad|>"), 4)
        self.assertIn("image_grid_thw", result)

    def test_action_head_uses_concatenated_input(self):
        head = ActionHead(hidden_size=2048, proprio_dim=25)
        first_linear = head.net[0]
        self.assertEqual(first_linear.in_features, 2074)
        output = head(torch.zeros(2, 2048), torch.zeros(2, 26))
        self.assertEqual(tuple(output.shape), (2, 7))
        self.assertTrue(bool((output.abs() <= 1).all()))

    def test_auxiliary_vector_pads_and_adds_progress(self):
        result = auxiliary_vector([1.0, 2.0], step=5, horizon=10, proprio_dim=4)
        np.testing.assert_allclose(result, [1.0, 2.0, 0.0, 0.0, 0.5])

    def test_last_token_pool_handles_left_and_right_padding(self):
        hidden = torch.arange(24).reshape(2, 4, 3)
        mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 0]])
        pooled = last_token_pool(hidden, mask)
        torch.testing.assert_close(pooled[0], hidden[0, 1])
        torch.testing.assert_close(pooled[1], hidden[1, 2])

    def test_multimodal_hidden_state_avoids_lm_head(self):
        qwen = FakeQwen(hidden_size=6)
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "pixel_values": torch.zeros((1, 3)),
        }
        hidden = qwen_last_hidden_state(qwen, inputs)
        self.assertEqual(tuple(hidden.shape), (1, 3, 6))

    def test_legacy_transformers_multimodal_path(self):
        qwen = LegacyQwen(hidden_size=4)
        inputs = {
            "input_ids": torch.tensor([[1, 99, 2]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "pixel_values": torch.zeros((1, 3)),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }
        hidden = qwen_last_hidden_state(qwen, inputs)
        self.assertEqual(tuple(hidden.shape), (1, 3, 4))
        torch.testing.assert_close(hidden[0, 1], torch.ones(4))

    def test_episode_split_never_leaks_an_episode(self):
        episode_hashes = torch.tensor(
            [
                stable_episode_hash("a"),
                stable_episode_hash("a"),
                stable_episode_hash("b"),
                stable_episode_hash("c"),
            ]
        )
        train, validation, train_episodes, validation_episodes = episode_split_masks(
            episode_hashes, val_fraction=0.34, seed=7
        )
        self.assertFalse(bool((train & validation).any()))
        self.assertTrue(bool(train.any()))
        self.assertTrue(bool(validation.any()))
        self.assertEqual(train_episodes + validation_episodes, 3)
        for episode_hash in set(episode_hashes.tolist()):
            indices = episode_hashes == episode_hash
            self.assertTrue(bool(train[indices].all() or validation[indices].all()))

    def test_cache_manifest_detects_semantic_mismatch(self):
        args = parse_args([])
        manifest = cache_spec(args)
        validate_cache_manifest(manifest, cache_spec(args))
        changed = dict(manifest)
        changed["image_size"] = 112
        with self.assertRaisesRegex(ValueError, "image_size"):
            validate_cache_manifest(changed, cache_spec(args))

    def test_adapter_returns_submission_shape_and_dtype(self):
        hidden_size = 8
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "qwen").mkdir()
            config = {
                "hidden_size": hidden_size,
                "proprio_dim": 25,
                "action_dim": 7,
                "image_size": 224,
                "attn_implementation": "sdpa",
                "backbone_refresh_interval": 2,
                "auxiliary_mean": [0.0] * 26,
                "auxiliary_std": [1.0] * 26,
            }
            (root / "policy_config.json").write_text(json.dumps(config))
            head = ActionHead(hidden_size=hidden_size, proprio_dim=25)
            save_file(
                {
                    key: value.detach().contiguous()
                    for key, value in head.state_dict().items()
                },
                root / "action_head.safetensors",
            )

            fake_qwen = FakeQwen(hidden_size)
            with (
                mock.patch(
                    "scripts.qwen_frozen_policy_adapter.load_qwen_processor",
                    return_value=FakeProcessor(),
                ),
                mock.patch(
                    "scripts.qwen_frozen_policy_adapter."
                    "Qwen2_5_VLForConditionalGeneration.from_pretrained",
                    return_value=fake_qwen,
                ),
            ):
                policy = FrozenQwenVLAPolicy(
                    model_dir=str(root),
                    device="cpu",
                    dtype="float32",
                )
                action = policy.act(
                    {
                        "image": np.zeros((96, 96, 3), dtype=np.uint8),
                        "instruction": "lift the cube",
                        "task": "lift_cube",
                        "difficulty": "low",
                        "proprio": np.zeros(25, dtype=np.float32),
                        "step": 10,
                        "horizon": 100,
                    }
                )
                policy.act(
                    {
                        "image": np.ones((96, 96, 3), dtype=np.uint8),
                        "instruction": "lift the cube",
                        "task": "lift_cube",
                        "difficulty": "low",
                        "proprio": np.ones(25, dtype=np.float32),
                        "step": 11,
                        "horizon": 100,
                    }
                )
                policy.act(
                    {
                        "image": np.ones((96, 96, 3), dtype=np.uint8),
                        "instruction": "lift the cube",
                        "task": "lift_cube",
                        "difficulty": "low",
                        "proprio": np.ones(25, dtype=np.float32),
                        "step": 12,
                        "horizon": 100,
                    }
                )
                self.assertEqual(fake_qwen.model.forward_calls, 2)

                policy.act(
                    {
                        "image": np.ones((96, 96, 3), dtype=np.uint8),
                        "instruction": "lift the cube",
                        "task": "lift_cube",
                        "difficulty": "low",
                        "proprio": np.ones(25, dtype=np.float32),
                        "step": 0,
                        "horizon": 100,
                    }
                )
                self.assertEqual(fake_qwen.model.forward_calls, 3)
                policy._embedding_for_step(
                    np.ones((96, 96, 3), dtype=np.uint8),
                    "changed prompt",
                    step=1,
                )
                self.assertEqual(fake_qwen.model.forward_calls, 4)
            self.assertEqual(action.shape, (7,))
            self.assertEqual(action.dtype, np.float32)
            self.assertTrue(np.isfinite(action).all())

    def test_end_to_end_pipeline_with_tiny_fakes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            (source / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen2_5_vl",
                        "hidden_size": 8,
                    }
                )
            )
            (source / "tokenizer_config.json").write_text("{}")
            (source / "preprocessor_config.json").write_text("{}")
            save_file(
                {"weight": torch.zeros(1)},
                source / "model.safetensors",
            )
            rows = []
            for index in range(6):
                rows.append(
                    {
                        "episode_id": f"episode-{index // 2}",
                        "task": "lift_cube",
                        "instruction": "lift the cube",
                        "difficulty": "low",
                        "step": index % 2,
                        "horizon": 2,
                        "image": np.zeros((96, 96, 3), dtype=np.uint8),
                        "proprio": np.zeros(25, dtype=np.float32),
                        "action": np.full(7, index / 10, dtype=np.float32),
                    }
                )
            output = root / "policy"
            cache = root / "cache"
            with (
                mock.patch(
                    "scripts.train_frozen_qwen_vla.load_backbone",
                    side_effect=lambda *args: (
                        FakeQwen(8),
                        FakeProcessor(),
                        "fake-sha",
                    ),
                ),
                mock.patch(
                    "scripts.train_frozen_qwen_vla.iter_dataset_rows",
                    return_value=iter(rows),
                ),
            ):
                summary = main(
                    [
                        "--model",
                        str(source),
                        "--dataset",
                        "fake",
                        "--out",
                        str(output),
                        "--cache-dir",
                        str(cache),
                        "--epochs",
                        "2",
                        "--batch-size",
                        "2",
                        "--embedding-batch-size",
                        "2",
                        "--cache-shard-samples",
                        "3",
                        "--device",
                        "cpu",
                        "--dtype",
                        "float32",
                        "--precompute-embeddings",
                    ]
                )
            self.assertEqual(summary["samples"], 6)
            self.assertTrue((output / "flock_robotics_adapter.py").is_file())
            self.assertTrue((output / "action_head.safetensors").is_file())
            self.assertTrue((output / "qwen" / "model.safetensors").is_file())
            policy_config = json.loads((output / "policy_config.json").read_text())
            self.assertEqual(policy_config["hidden_size"], 8)
            self.assertEqual(policy_config["backbone_refresh_interval"], 2)


if __name__ == "__main__":
    unittest.main()
