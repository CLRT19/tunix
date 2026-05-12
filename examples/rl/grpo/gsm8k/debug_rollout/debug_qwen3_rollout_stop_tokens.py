#!/usr/bin/env python3
"""Standalone Qwen3 rollout stop-token probe.

This mirrors the vanilla rollout path used by the Qwen3 GSM8K GRPO scripts:

1. Builds `GrpoPipeline.create_rollout_config()` from Qwen-style overrides and
   verifies the rollout config defaults to `[151643, 151645]`.
2. Runs Tunix's vanilla sampler twice on the same prompt:
   - control: stop only on tokenizer.eos_id() (`151643` for Qwen)
   - test: stop on the Qwen rollout config's eos_tokens (`151643,151645`)

The probe passes when the control emits `<|im_end|>` and the test output stops
before that token. This catches the exact failure mode where `<|im_end|>` is
treated as ordinary text and generation continues.
"""

from __future__ import annotations

import argparse
import os
import socket
from typing import Sequence


def _set_default_env() -> None:
  os.environ.setdefault("JAX_PLATFORMS", "cpu")
  os.environ.setdefault("TUNIX_JAX_DISTRIBUTED_AUTO_INIT", "0")


_set_default_env()

from flax import nnx  # pylint: disable=g-import-not-at-top
import jax  # pylint: disable=g-import-not-at-top
import numpy as np  # pylint: disable=g-import-not-at-top
from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top

from tunix.cli import grpo_main  # pylint: disable=g-import-not-at-top
from tunix.generate import sampler as tunix_sampler  # pylint: disable=g-import-not-at-top
from tunix.generate import tokenizer_adapter  # pylint: disable=g-import-not-at-top
from tunix.models import automodel  # pylint: disable=g-import-not-at-top


_QWEN_STOP_TOKENS = [151643, 151645]
_IM_END_TOKEN_ID = 151645
_IM_END_TOKEN = "<|im_end|>"
_DEFAULT_PROMPTS = (
    "Answer exactly in this format: <think>2+2=4</think><answer>4</answer>",
    "What is 2+2? Think briefly, give <answer>4</answer>, then end.",
    "For a diagnostic, complete the assistant turn with a short math answer.",
)


def _maybe_initialize_jax_distributed() -> None:
  if os.environ.get("TUNIX_JAX_DISTRIBUTED_AUTO_INIT", "").lower() not in (
      "1",
      "true",
      "yes",
  ):
    return
  if jax.distributed.is_initialized():
    return
  timeout_seconds = int(
      os.environ.get("TUNIX_JAX_DISTRIBUTED_INIT_TIMEOUT_SECONDS", "300")
  )
  print(
      "Initializing JAX distributed runtime "
      f"(timeout={timeout_seconds}s)...",
      flush=True,
  )
  jax.distributed.initialize(initialization_timeout=timeout_seconds)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-name", default="Qwen3-1.7B-base")
  parser.add_argument("--max-new-tokens", type=int, default=512)
  parser.add_argument("--max-prompt-length", type=int, default=256)
  parser.add_argument("--cache-size", type=int, default=None)
  parser.add_argument("--temperature", type=float, default=0.9)
  parser.add_argument("--top-p", type=float, default=1.0)
  parser.add_argument("--top-k", type=int, default=50)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
      "--mesh-shape",
      default=None,
      help=(
          "Mesh shape as fsdp,tp. Defaults to 8,4 for 32 TPU devices and"
          " 1,<device_count> otherwise."
      ),
  )
  parser.add_argument(
      "--prompt",
      action="append",
      default=None,
      help="Prompt to test. Can be repeated. Defaults to built-in prompts.",
  )
  parser.add_argument(
      "--allow-inconclusive",
      action="store_true",
      help=(
          "Do not fail if the control generation never emits <|im_end|>."
          " The config default is still verified."
      ),
  )
  parser.add_argument(
      "--model-download-path",
      default=None,
      help=(
          "Local directory for Tunix's HF download. Defaults to"
          " /tmp/tunix_debug_models/<model-name>."
      ),
  )
  return parser.parse_args()


def _make_mesh(mesh_shape_arg: str | None) -> jax.sharding.Mesh:
  if mesh_shape_arg:
    mesh_shape = tuple(int(x.strip()) for x in mesh_shape_arg.split(","))
  elif jax.process_count() > 1 and jax.device_count() == 32:
    mesh_shape = (8, 4)
  else:
    mesh_shape = (1, jax.device_count())
  if len(mesh_shape) != 2:
    raise ValueError(f"--mesh-shape must have two dimensions: {mesh_shape_arg}")
  devices = np.asarray(jax.devices(), dtype=object).reshape(mesh_shape)
  print(f"mesh_shape={mesh_shape}", flush=True)
  return jax.sharding.Mesh(devices, ("fsdp", "tp"))


def _format_prompt(hf_tokenizer: AutoTokenizer, prompt: str) -> str:
  system_prompt = (
      "You are given a problem. Think about the problem and provide your"
      " reasoning. Place it between <think> and </think>. Then, provide"
      " the final answer (i.e., just one numerical value) between <answer>"
      " and </answer>."
  )
  return hf_tokenizer.apply_chat_template(
      [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": prompt},
      ],
      tokenize=False,
      add_generation_prompt=True,
      enable_thinking=True,
  )


def _create_rollout_config(model_name: str) -> object:
  model_id = f"Qwen/{model_name}"
  argv = [
      "grpo_main",
      "base_config.yaml",
      f"model_config.model_name={model_name}",
      f"model_config.model_id={model_id}",
      "model_config.model_source=huggingface",
      f"model_config.intermediate_ckpt_dir=/tmp/intermediate_ckpt/{model_name}",
      f"actor_model_config.model_id={model_id}",
      f"reference_model_config.model_id={model_id}",
      f"rollout_model_config.model_id={model_id}",
      f"tokenizer_config.tokenizer_path={model_id}",
      f"rollout_config.rollout_vllm_model_version={model_id}",
  ]
  return grpo_main.GrpoPipeline(argv).create_rollout_config()


def _completion_tokens(output, tokenizer) -> list[int]:
  tokens = np.asarray(output.tokens[0], dtype=np.int64).reshape(-1)
  return [int(x) for x in tokens if int(x) != tokenizer.pad_id()]


def _decode(tokenizer, token_ids: Sequence[int]) -> str:
  return tokenizer.decode(list(token_ids), skip_special_tokens=False)


def _first_index(values: Sequence[int], target: int) -> int | None:
  for i, value in enumerate(values):
    if value == target:
      return i
  return None


def main() -> None:
  args = _parse_args()
  _maybe_initialize_jax_distributed()

  model_id = f"Qwen/{args.model_name}"
  model_download_path = args.model_download_path or os.path.join(
      "/tmp", "tunix_debug_models", args.model_name
  )

  rollout_config = _create_rollout_config(args.model_name)
  print("ENV:")
  print(f"  hostname={socket.gethostname()}")
  print(f"  jax.default_backend()={jax.default_backend()}")
  print(f"  jax.process_index()={jax.process_index()}")
  print(f"  jax.process_count()={jax.process_count()}")
  print(f"  jax.device_count()={jax.device_count()}")
  print(f"  model_id={model_id}")
  print(f"  rollout_config.eos_tokens={rollout_config.eos_tokens}")

  if rollout_config.eos_tokens != _QWEN_STOP_TOKENS:
    raise AssertionError(
        "Qwen rollout config did not default to expected stop tokens: "
        f"got {rollout_config.eos_tokens}, expected {_QWEN_STOP_TOKENS}"
    )

  hf_token = os.environ.get("HF_TOKEN")
  hf_tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
  tokenizer = tokenizer_adapter.Tokenizer(
      tokenizer_type="huggingface",
      tokenizer_path=model_id,
      add_bos=False,
      add_eos=False,
      hf_access_token=hf_token,
  )

  print("\nLoading Tunix model...")
  mesh = _make_mesh(args.mesh_shape)
  model, resolved_model_path = automodel.AutoModel.from_pretrained(
      model_id=model_id,
      mesh=mesh,
      model_source=automodel.ModelSource.HUGGINGFACE,
      model_download_path=model_download_path,
  )
  print(f"resolved_model_path={resolved_model_path}")
  if not hasattr(model, "num_embed") and hasattr(model.config, "vocab_size"):
    model.num_embed = model.config.vocab_size

  cache_size = args.cache_size or (
      args.max_prompt_length + args.max_new_tokens + 256
  )
  sampler = tunix_sampler.Sampler(
      model,
      tokenizer,
      tunix_sampler.CacheConfig(
          cache_size=cache_size,
          num_layers=model.config.num_layers,
          num_kv_heads=model.config.num_kv_heads,
          head_dim=model.config.head_dim,
      ),
  )

  prompts = args.prompt or list(_DEFAULT_PROMPTS)
  prompt = prompts[jax.process_index() % len(prompts)]
  chat_text = _format_prompt(hf_tokenizer, prompt)
  print(f"\nPROMPT={prompt!r}")
  print(f"CHAT_TEXT_TAIL={chat_text[-240:]!r}")

  with mesh:
    control_output = sampler(
        input_strings=[chat_text],
        max_generation_steps=args.max_new_tokens,
        max_prompt_length=args.max_prompt_length,
        echo=False,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        pad_output=True,
        eos_tokens=[tokenizer.eos_id()],
    )
    test_output = sampler(
        input_strings=[chat_text],
        max_generation_steps=args.max_new_tokens,
        max_prompt_length=args.max_prompt_length,
        echo=False,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        pad_output=True,
        eos_tokens=list(rollout_config.eos_tokens),
    )

  control_tokens = _completion_tokens(control_output, tokenizer)
  test_tokens = _completion_tokens(test_output, tokenizer)
  control_text = _decode(tokenizer, control_tokens)
  test_text = _decode(tokenizer, test_tokens)
  control_im_end_index = _first_index(control_tokens, _IM_END_TOKEN_ID)

  print("\nCONTROL_EOS_ONLY:")
  print(f"  token_count={len(control_tokens)}")
  print(f"  contains_im_end={_IM_END_TOKEN in control_text}")
  print(f"  contains_im_end_token={control_im_end_index is not None}")
  print(f"  tail={control_text[-500:]!r}")
  print("\nTEST_QWEN_STOP_TOKENS:")
  print(f"  token_count={len(test_tokens)}")
  print(f"  contains_im_end={_IM_END_TOKEN in test_text}")
  print(f"  contains_im_end_token={_IM_END_TOKEN_ID in test_tokens}")
  print(f"  hit_generation_cap={len(test_tokens) >= args.max_new_tokens}")
  print(f"  text={test_text!r}")

  if control_im_end_index is None:
    message = (
        "Control generation never emitted <|im_end|>; stop-token behavior is"
        " inconclusive for this prompt/model/seed."
    )
    if args.allow_inconclusive:
      print(f"\nINCONCLUSIVE: {message}")
      return
    raise AssertionError(message)
  if _IM_END_TOKEN_ID in test_tokens:
    raise AssertionError(
        "Qwen stop-token generation leaked <|im_end|> into returned tokens."
    )
  if len(test_tokens) >= args.max_new_tokens:
    raise AssertionError(
        "Qwen stop-token generation hit max_new_tokens instead of stopping."
    )
  expected_prefix = control_tokens[:control_im_end_index]
  if test_tokens != expected_prefix:
    raise AssertionError(
        "Qwen stop-token generation did not match the EOS-only control prefix"
        " before <|im_end|>."
    )

  print(
      "\nPASS: rollout_config defaulted to [151643,151645] and sampler"
      " stopped before <|im_end|>."
  )


if __name__ == "__main__":
  main()
