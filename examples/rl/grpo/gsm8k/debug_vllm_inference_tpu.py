"""Standalone vLLM TPU inference probe.

This intentionally avoids GRPO, checkpointing, and Tunix RL cluster setup. It
only initializes JAX, builds a small local TPU mesh, starts Tunix's vLLM sampler,
and runs one short generation request.
"""

from __future__ import annotations

import ast
import os
import socket
import sys
import time
import traceback

import jax
import numpy as np

from tunix.generate import tokenizer_adapter
from tunix.generate import vllm_sampler


def _worker_id_from_hostname(host: str) -> str:
  marker = "-w-"
  if marker not in host:
    return ""
  return host.rsplit(marker, 1)[-1].split(".", 1)[0]


def _get_bool(name: str, default: bool) -> bool:
  value = os.environ.get(name)
  if value is None:
    return default
  return value.lower() in ("1", "true", "yes", "on")


def _get_int(name: str, default: int) -> int:
  return int(os.environ.get(name, str(default)))


def _get_float(name: str, default: float) -> float:
  return float(os.environ.get(name, str(default)))


def _get_tuple(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
  value = os.environ.get(name)
  if value is None:
    return default
  parsed = ast.literal_eval(value)
  if not isinstance(parsed, tuple) or not all(isinstance(x, int) for x in parsed):
    raise ValueError(f"{name} must be a tuple of ints, got {value!r}")
  return parsed


def _make_mesh() -> jax.sharding.Mesh:
  mesh_shape = _get_tuple("VLLM_DEBUG_MESH_SHAPE", (1, 4))
  axis_names = ast.literal_eval(
      os.environ.get("VLLM_DEBUG_MESH_AXIS_NAMES", "('dp', 'tp')")
  )
  if not isinstance(axis_names, tuple) or not all(
      isinstance(x, str) for x in axis_names
  ):
    raise ValueError("VLLM_DEBUG_MESH_AXIS_NAMES must be a tuple of strings.")
  if len(axis_names) != len(mesh_shape):
    raise ValueError(
        "VLLM_DEBUG_MESH_AXIS_NAMES length must match VLLM_DEBUG_MESH_SHAPE."
    )

  mesh_scope = os.environ.get("VLLM_DEBUG_MESH_SCOPE", "local")
  devices = jax.local_devices() if mesh_scope == "local" else jax.devices()
  needed = int(np.prod(mesh_shape))
  if len(devices) < needed:
    raise ValueError(
        f"Need {needed} devices for mesh shape {mesh_shape}, but only found "
        f"{len(devices)} {mesh_scope} devices: {devices}"
    )
  mesh_devices = np.array(devices[:needed], dtype=object).reshape(mesh_shape)
  return jax.sharding.Mesh(mesh_devices, axis_names)


def main() -> None:
  start = time.time()
  host = socket.gethostname()
  print(f"[vllm-debug] host={host}", flush=True)
  backend = jax.default_backend()
  process_index = jax.process_index()
  process_count = jax.process_count()
  device_count = jax.device_count()
  local_device_count = jax.local_device_count()
  print(
      "[vllm-debug] jax "
      f"backend={backend} "
      f"process={process_index}/{process_count} "
      f"devices={device_count} local_devices={local_device_count}",
      flush=True,
  )
  active_worker = os.environ.get("VLLM_DEBUG_ACTIVE_WORKER", "all")
  worker_id = _worker_id_from_hostname(host)
  worker_selector = worker_id or str(process_index)
  if active_worker != "all" and active_worker != worker_selector:
    hold_seconds = _get_int("VLLM_DEBUG_PARTICIPANT_HOLD_SECONDS", 180)
    print(
        "[vllm-debug] participant holding without vLLM "
        f"worker_id={worker_id or 'unknown'} process_index={process_index} "
        f"active_worker={active_worker} "
        f"hold_seconds={hold_seconds}",
        flush=True,
    )
    time.sleep(hold_seconds)
    print("[vllm-debug] participant done", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    # Participants only exist so multi-host JAX initialization can form the
    # slice. Do not let TPU/JAX teardown noise turn a passive participant into a
    # failed active-worker probe.
    os._exit(0)  # pylint: disable=protected-access

  model_id = os.environ.get("VLLM_DEBUG_MODEL_ID")
  if not model_id:
    model_name = os.environ.get("model_name", "Qwen3-1.7B-base")
    model_id = f"Qwen/{model_name}"

  prompt = os.environ.get("VLLM_DEBUG_PROMPT", "Question: What is 2 + 2?\nAnswer:")
  max_model_len = _get_int("VLLM_DEBUG_MAX_MODEL_LEN", 128)
  max_num_seqs = _get_int("VLLM_DEBUG_MAX_NUM_SEQS", 1)
  max_num_batched_tokens = _get_int("VLLM_DEBUG_MAX_NUM_BATCHED_TOKENS", 128)
  max_generation_steps = _get_int("VLLM_DEBUG_MAX_GENERATION_STEPS", 16)
  max_prompt_length = _get_int("VLLM_DEBUG_MAX_PROMPT_LENGTH", 64)
  tensor_parallel_size = _get_int("VLLM_DEBUG_TENSOR_PARALLEL_SIZE", 4)
  data_parallel_size = _get_int("VLLM_DEBUG_DATA_PARALLEL_SIZE", 1)
  hbm_utilization = _get_float("VLLM_DEBUG_HBM_UTILIZATION", 0.2)
  temperature = _get_float("VLLM_DEBUG_TEMPERATURE", 0.0)
  init_with_random_weights = _get_bool("VLLM_DEBUG_INIT_RANDOM_WEIGHTS", True)
  server_mode = _get_bool("VLLM_DEBUG_SERVER_MODE", False)

  if max_num_batched_tokens > max_num_seqs * max_model_len:
    raise ValueError(
        "Invalid tiny vLLM config: VLLM_DEBUG_MAX_NUM_BATCHED_TOKENS "
        f"({max_num_batched_tokens}) exceeds VLLM_DEBUG_MAX_NUM_SEQS * "
        f"VLLM_DEBUG_MAX_MODEL_LEN ({max_num_seqs * max_model_len})."
    )

  os.environ.setdefault("TUNIX_VLLM_USE_LOCAL_TPU_DEVICE_IDS", "1")
  mesh = _make_mesh()
  print(
      "[vllm-debug] config "
      f"active_worker={active_worker} worker_selector={worker_selector} "
      f"model_id={model_id} mesh_shape={mesh.devices.shape} "
      f"mesh_axis_names={mesh.axis_names} tp={tensor_parallel_size} "
      f"dp={data_parallel_size} max_model_len={max_model_len} "
      f"max_num_seqs={max_num_seqs} "
      f"max_num_batched_tokens={max_num_batched_tokens} "
      f"max_generation_steps={max_generation_steps} "
      f"init_with_random_weights={init_with_random_weights} "
      f"server_mode={server_mode}",
      flush=True,
  )

  tokenizer = tokenizer_adapter.Tokenizer(
      tokenizer_type="huggingface",
      tokenizer_path=model_id,
      add_bos=False,
      add_eos=False,
      hf_access_token=os.environ.get("HF_TOKEN"),
  )
  prompt_tokens = tokenizer.encode(prompt)
  print(
      f"[vllm-debug] prompt_token_count={len(prompt_tokens)} prompt={prompt!r}",
      flush=True,
  )
  if len(prompt_tokens) > max_prompt_length:
    raise ValueError(
        f"Prompt has {len(prompt_tokens)} tokens, which exceeds "
        f"VLLM_DEBUG_MAX_PROMPT_LENGTH={max_prompt_length}."
    )

  config = vllm_sampler.VllmConfig(
      server_mode=server_mode,
      init_with_random_weights=init_with_random_weights,
      tpu_backend_type=os.environ.get("VLLM_DEBUG_TPU_BACKEND_TYPE", "jax"),
      hbm_utilization=hbm_utilization,
      mesh=mesh,
      data_parallel_size=data_parallel_size,
      tensor_parallel_size=tensor_parallel_size,
      engine_kwargs={
          "model": model_id,
          "max_model_len": max_model_len,
          "max_num_seqs": max_num_seqs,
          "max_num_batched_tokens": max_num_batched_tokens,
          "swap_space": _get_float("VLLM_DEBUG_SWAP_SPACE_GB", 4.0),
      },
  )

  print("[vllm-debug] constructing VllmSampler", flush=True)
  sampler = vllm_sampler.VllmSampler(tokenizer=tokenizer, config=config)
  print("[vllm-debug] VllmSampler constructed; generating", flush=True)
  output = sampler(
      prompt,
      max_generation_steps=max_generation_steps,
      max_prompt_length=max_prompt_length,
      temperature=temperature,
      return_logits=False,
  )
  print(f"[vllm-debug] output_text={output.text}", flush=True)
  print(
      f"[vllm-debug] output_token_lengths={[len(x) for x in output.tokens]}",
      flush=True,
  )
  sampler.stop()
  print(
      f"[vllm-debug] success elapsed_seconds={time.time() - start:.2f}",
      flush=True,
  )
  print(
      "[vllm-debug] result "
      f"active_worker={active_worker} worker_id={worker_selector} "
      "status=success",
      flush=True,
  )
  sys.stdout.flush()
  sys.stderr.flush()
  # The TPU backend may still run process teardown hooks after the probe has
  # already proven initialization and one generation succeeded. Exit directly so
  # this debug script reports the probe result, not shutdown cleanup behavior.
  os._exit(0)  # pylint: disable=protected-access


if __name__ == "__main__":
  try:
    main()
  except Exception:  # pylint: disable=broad-exception-caught
    worker_selector = _worker_id_from_hostname(socket.gethostname())
    print("[vllm-debug] failed with exception:", flush=True)
    print(
        "[vllm-debug] result "
        f"active_worker={os.environ.get('VLLM_DEBUG_ACTIVE_WORKER', 'all')} "
        f"worker_id={worker_selector or 'unknown'} "
        "status=failure",
        flush=True,
    )
    traceback.print_exc()
    raise
