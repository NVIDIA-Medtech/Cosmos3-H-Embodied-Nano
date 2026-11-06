"""Exercise the real public argument builder with in-memory H16 conditioning."""
import json
from types import SimpleNamespace

import pytest
import torch
from cosmos_framework.configs.base.defaults.model_config import FixedStepSamplerConfig
from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModelConfig

# The executable evaluator initializes inference and disables autograd on import.
# Restore the caller's grad mode so unrelated training-contract tests are intact.
with torch.enable_grad():
    from cosmos3_h_embodied.dreams.evaluate import build_dataset_sample_args


@pytest.mark.parametrize("steps", [None, 2, 4])
def test_dataset_forward_dynamics_args_keep_history_and_causal_dispatch(tmp_path, steps):
    cfg = OmniMoTCausalModelConfig(action_gen=True, video_temporal_causal=steps is not None)
    cfg.tokenizer = SimpleNamespace(temporal_compression_factor=4)
    if steps is not None:
        cfg.fixed_step_sampler_config = FixedStepSamplerConfig(
            t_list=[1, 5 / 6] if steps == 2 else [1, 0.9375, 5 / 6, 0.625], sample_type="sde"
        )
    action = torch.arange(88 * 64, dtype=torch.float32).reshape(88, 64)
    original_action = action.clone()
    item = {
        "action": action,
        "sequence_plan": SimpleNamespace(action_start_frame_offset=-15),
        "__key__": "sentinel_h16_c72",
        "ai_caption": "Continue the ongoing surgical task.",
    }
    args = build_dataset_sample_args(
        item, cfg, name="recorded", destination=tmp_path, future=72, seed=28003
    )
    assert args.model_mode.value == "forward_dynamics"
    assert args.autoregressive is (steps is not None)
    assert args.num_steps == (steps or 35)
    assert args.num_frames == 73 and args.fps == 10
    assert args.raw_action_dim == 28
    restored = torch.tensor(json.loads(args.action_path.read_text()))
    torch.testing.assert_close(restored, action[:, :28], rtol=0, atol=0)
    metadata = json.loads((tmp_path / "conditioning_metadata.json").read_text())
    assert metadata["history"] == 16 and metadata["action_start_frame_offset"] == -15
    assert metadata["future_frames"] == 72
    torch.testing.assert_close(item["action"], original_action, rtol=0, atol=0)
