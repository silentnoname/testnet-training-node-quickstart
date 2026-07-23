from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration


ACTION_DIM = 7
DEFAULT_PROPRIO_DIM = 25
PARAMETER_LIMIT = 4_500_000_000
PROMPT_VERSION = "frozen_qwen_vla_v1"

SYSTEM_PROMPT = (
    "You are the visual-language backbone of a low-level controller for a "
    "7-DOF Franka Panda robot. Encode the current image and requested "
    "manipulation so a separate action head can predict the next Cartesian "
    "delta and gripper command."
)


class ImageOnlyQwenProcessor:
    """Qwen processor subset that avoids an unnecessary video dependency."""

    def __init__(self, image_processor: Any, tokenizer: Any) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.image_token = getattr(tokenizer, "image_token", "<|image_pad|>")

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> str:
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def __call__(
        self,
        *,
        images: list[Image.Image],
        text: list[str],
        padding: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        image_inputs = self.image_processor(
            images=images,
            return_tensors=return_tensors,
        )
        image_grid_thw = image_inputs["image_grid_thw"]
        merge_length = int(self.image_processor.merge_size) ** 2
        expanded_text = list(text)
        image_index = 0
        for text_index, value in enumerate(expanded_text):
            while self.image_token in value:
                if image_index >= len(image_grid_thw):
                    raise ValueError("Prompt contains more image tokens than images.")
                repeat_count = int(
                    image_grid_thw[image_index].prod().item() // merge_length
                )
                value = value.replace(
                    self.image_token,
                    "<|flock_image_placeholder|>" * repeat_count,
                    1,
                )
                image_index += 1
            expanded_text[text_index] = value.replace(
                "<|flock_image_placeholder|>",
                self.image_token,
            )
        if image_index != len(image_grid_thw):
            raise ValueError("Processor received more images than prompt image tokens.")
        text_inputs = self.tokenizer(
            expanded_text,
            padding=padding,
            return_tensors=return_tensors,
        )
        return {**text_inputs, **image_inputs}


def load_qwen_processor(
    model_name_or_path: str | Path,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
) -> ImageOnlyQwenProcessor:
    from transformers import Qwen2VLImageProcessor

    kwargs: dict[str, Any] = {"local_files_only": local_files_only}
    if revision is not None:
        kwargs["revision"] = revision
    image_processor = Qwen2VLImageProcessor.from_pretrained(
        model_name_or_path,
        **kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
    return ImageOnlyQwenProcessor(image_processor, tokenizer)


class ActionHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        proprio_dim: int = DEFAULT_PROPRIO_DIM,
        action_dim: int = ACTION_DIM,
    ) -> None:
        super().__init__()
        input_dim = hidden_size + proprio_dim + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def forward(
        self, qwen_embedding: torch.Tensor, auxiliary: torch.Tensor
    ) -> torch.Tensor:
        return self.net(torch.cat((qwen_embedding, auxiliary), dim=-1))


def build_prompt(
    processor: Any,
    instruction: str,
    task: str = "",
    difficulty: str = "",
) -> str:
    difficulty_text = difficulty if difficulty and difficulty != "None" else "unknown"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        f"Task: {task or 'robot manipulation'}\n"
                        f"Difficulty: {difficulty_text}\n"
                        f"Instruction: {instruction}\n"
                        "Encode the scene and intent for the next low-level action."
                    ),
                },
            ],
        },
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def auxiliary_vector(
    proprio: Any,
    step: int | float,
    horizon: int | float,
    proprio_dim: int = DEFAULT_PROPRIO_DIM,
) -> np.ndarray:
    values = np.asarray(proprio, dtype=np.float32).reshape(-1)
    if values.size < proprio_dim:
        values = np.pad(values, (0, proprio_dim - values.size))
    values = np.nan_to_num(
        values[:proprio_dim],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    safe_horizon = max(float(horizon or 1), 1.0)
    progress = np.asarray(
        [np.clip(float(step) / safe_horizon, 0.0, 1.0)], dtype=np.float32
    )
    return np.concatenate((values, progress), axis=0).astype(np.float32)


def last_token_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    positions = torch.arange(
        attention_mask.shape[1], device=attention_mask.device
    ).unsqueeze(0)
    last_positions = positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
    if bool((last_positions < 0).any()):
        raise ValueError("Cannot pool a sequence with an empty attention mask.")
    batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_indices, last_positions]


def qwen_last_hidden_state(
    qwen: Qwen2_5_VLForConditionalGeneration,
    model_inputs: dict[str, Any],
) -> torch.Tensor:
    """Run the multimodal base without allocating vocabulary-sized logits."""
    base_model = qwen.model
    if hasattr(base_model, "visual"):
        outputs = base_model(
            **model_inputs,
            use_cache=False,
            return_dict=True,
        )
        return outputs.last_hidden_state

    # Transformers 4.49 keeps visual integration on the conditional-generation
    # wrapper, while its `.model` is text-only. Reproduce the pre-lm_head part
    # of that forward pass so caching does not allocate [B, T, vocab] logits.
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask")
    image_grid_thw = model_inputs.get("image_grid_thw")
    pixel_values = model_inputs.get("pixel_values")
    inputs_embeds = base_model.embed_tokens(input_ids)
    if pixel_values is not None:
        visual_dtype = getattr(qwen.visual, "dtype", inputs_embeds.dtype)
        image_embeds = qwen.visual(
            pixel_values.type(visual_dtype),
            grid_thw=image_grid_thw,
        )
        image_mask = (input_ids == qwen.config.image_token_id).unsqueeze(-1)
        image_mask = image_mask.expand_as(inputs_embeds)
        if int(image_mask[..., 0].sum()) != int(image_embeds.shape[0]):
            raise ValueError("Qwen image tokens and visual features do not match.")
        inputs_embeds = inputs_embeds.masked_scatter(
            image_mask.to(inputs_embeds.device),
            image_embeds.to(inputs_embeds.device, inputs_embeds.dtype),
        )
    position_ids, _ = qwen.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )
    outputs = base_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        use_cache=False,
        return_dict=True,
    )
    return outputs.last_hidden_state


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    if (
        requested == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    normalized = str(name).lower().replace("torch.", "")
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized in aliases:
        resolved = aliases[normalized]
        if device.type == "mps" and resolved == torch.bfloat16:
            return torch.float16
        if device.type == "cpu" and resolved == torch.float16:
            return torch.float32
        return resolved
    if normalized not in {"", "auto", "none"}:
        raise ValueError(f"Unsupported dtype: {name}")
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def prepare_image(value: Any, image_size: int) -> Image.Image:
    array = np.asarray(value)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3:
        raise ValueError(f"Expected an HWC image, got shape {array.shape}.")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] > 3:
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        if array.size and float(array.max()) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    image = Image.fromarray(array).convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize(
            (image_size, image_size),
            resample=Image.Resampling.BILINEAR,
        )
    return image


class FrozenQwenVLAPolicy(nn.Module):
    def __init__(self, model_dir: str, device: str, dtype: str) -> None:
        super().__init__()
        root = Path(model_dir)
        self.config = json.loads((root / "policy_config.json").read_text())
        self.device = resolve_device(device)
        self.backbone_dtype = resolve_dtype(dtype, self.device)
        self.image_size = int(self.config["image_size"])
        self.proprio_dim = int(self.config["proprio_dim"])

        qwen_dir = root / "qwen"
        self.processor = load_qwen_processor(
            qwen_dir,
            local_files_only=True,
        )
        self.qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_dir,
            torch_dtype=self.backbone_dtype,
            attn_implementation=self.config.get("attn_implementation", "sdpa"),
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.qwen.requires_grad_(False)
        self.qwen.eval()

        self.action_head = ActionHead(
            hidden_size=int(self.config["hidden_size"]),
            proprio_dim=self.proprio_dim,
            action_dim=int(self.config.get("action_dim", ACTION_DIM)),
        )
        state_dict = load_file(str(root / "action_head.safetensors"), device="cpu")
        self.action_head.load_state_dict(state_dict, strict=True)
        self.action_head.to(device=self.device, dtype=torch.float32)
        self.action_head.requires_grad_(False)
        self.action_head.eval()

        auxiliary_mean = torch.tensor(
            self.config["auxiliary_mean"], dtype=torch.float32
        )
        auxiliary_std = torch.tensor(
            self.config["auxiliary_std"], dtype=torch.float32
        ).clamp_min(1e-6)
        self.register_buffer("auxiliary_mean", auxiliary_mean.to(self.device))
        self.register_buffer("auxiliary_std", auxiliary_std.to(self.device))

        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        if parameter_count > PARAMETER_LIMIT:
            raise ValueError(
                f"Policy has {parameter_count:,} parameters; limit is "
                f"{PARAMETER_LIMIT:,}."
            )

    @lru_cache(maxsize=64)
    def _cached_prompt(self, instruction: str, task: str, difficulty: str) -> str:
        return build_prompt(self.processor, instruction, task, difficulty)

    def _encode(self, image: Image.Image, prompt: str) -> torch.Tensor:
        inputs = self.processor(
            text=[prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        return last_token_pool(
            qwen_last_hidden_state(self.qwen, inputs),
            inputs["attention_mask"],
        ).float()

    @torch.inference_mode()
    def act(self, obs: dict[str, Any]) -> np.ndarray:
        image = prepare_image(obs["image"], self.image_size)
        instruction = str(obs.get("instruction", ""))
        task = str(obs.get("task", ""))
        difficulty = str(obs.get("difficulty", "") or "")
        prompt = self._cached_prompt(instruction, task, difficulty)
        embedding = self._encode(image, prompt)

        auxiliary = auxiliary_vector(
            obs.get("proprio", np.zeros(self.proprio_dim, dtype=np.float32)),
            obs.get("step", 0),
            obs.get("horizon", 1),
            proprio_dim=self.proprio_dim,
        )
        auxiliary_tensor = torch.from_numpy(auxiliary).to(self.device).unsqueeze(0)
        auxiliary_tensor = (auxiliary_tensor - self.auxiliary_mean) / self.auxiliary_std
        action = self.action_head(embedding, auxiliary_tensor)
        result = action.squeeze(0).float().cpu().numpy()
        return np.clip(result, -1.0, 1.0).astype(np.float32)


def load_policy(model_dir: str, device: str, dtype: str) -> FrozenQwenVLAPolicy:
    return FrozenQwenVLAPolicy(model_dir=model_dir, device=device, dtype=dtype)
