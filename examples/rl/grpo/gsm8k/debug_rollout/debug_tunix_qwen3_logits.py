#!/usr/bin/env python3
"""Standalone Tunix Qwen3 rollout/logits probe.

This intentionally avoids GRPO. It loads the Tunix Qwen3 model, applies the
same HF chat template used by the GRPO script, runs Tunix's vanilla sampler,
and prints the first-step top logits plus a short greedy continuation.
"""

from __future__ import annotations

import argparse
import os
import socket
from typing import Sequence


def _set_default_env() -> None:
  # Keep the first probe off TPU so a single SSH'd worker can run it without
  # requiring all TPU hosts to join libtpu/JAX distributed initialization.
  os.environ.setdefault("JAX_PLATFORMS", "cpu")
  os.environ.setdefault("TUNIX_JAX_DISTRIBUTED_AUTO_INIT", "0")


_set_default_env()

from flax import nnx  # pylint: disable=g-import-not-at-top
import jax  # pylint: disable=g-import-not-at-top
import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
import numpy as np  # pylint: disable=g-import-not-at-top
from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top

from tunix.generate import sampler as tunix_sampler  # pylint: disable=g-import-not-at-top
from tunix.generate import tokenizer_adapter  # pylint: disable=g-import-not-at-top
from tunix.models import automodel  # pylint: disable=g-import-not-at-top


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


# Per-process prompts to exercise the multi-host sampler with DIFFERENT data
# per process (mirrors what GRPO multi-host actually does: each JAX process
# tokenizes its own GSM8K question). This is the test the GRPO trajectory
# investigation needs: if the bug is in shard_input/Sampler multi-host data
# routing rather than chat templating, then with 8 distinct per-process
# prompts we expect to see corruption (e.g. pairs of processes producing the
# same completion despite having different prompts).
_PER_PROCESS_PROMPTS: tuple[str, ...] = (
    "Jane is painting her fingernails. She applies a base coat that takes 2"
    " minutes to dry, two color coats that take 3 minutes each to dry, and a"
    " clear top coat that takes 5 minutes to dry. How many minutes total does"
    " Jane spend waiting for her nail polish to dry?",
    "Carmen had 28 cats and 18 dogs before she gave 3 of the cats up for"
    " adoption. How many more cats than dogs does Carmen have now?",
    "Henry scored 50 points on his Geography test, 70 on his Math test and 66"
    " on his English test. If his History score is the average of these"
    " scores, what is his total score across all 4 subjects?",
    "A store has an 8% discount on all items. If Shara paid $184 for a pair"
    " of shoes, how much did Shara save?",
    "Will and Henry go fishing in a river. Will catches 16 catfish and 10"
    " eels. Henry challenges himself to catch 3 trout for every catfish that"
    " Will catches. Due to environmental concerns, Henry decides to return"
    " half his catch after meeting his own challenge. How many fishes do they"
    " have altogether now?",
    "Daisy is a poodle puppy who loves to play with her dog toys. She often"
    " loses them in various ways, and her owner needs to replace them. On"
    " Monday, Daisy played with 5 dog toys. On Tuesday, Daisy had 3 dog toys"
    " left after losing some, and her owner went and bought her 3 more. On"
    " Wednesday, all of Daisy's old and new dog toys were missing, so her"
    " owner went to the store and bought her 5 more. If Daisy's owner found"
    " all the lost dog toys, including the new dog toys, how many dog toys"
    " would Daisy have now?",
    "Nathan planted 5 strawberry plants and 7 tomato plants. He harvested 14"
    " strawberries from each plant and 16 tomatoes from each plant. He then"
    " distributed the strawberries and tomatoes into baskets of 7. How many"
    " baskets of fruit did Nathan fill?",
    "Last night, Jim bought a $7 lamp and a bulb which cost $4 less. If he"
    " bought 2 lamps and 6 bulbs, how much did Jim pay in all?",
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-name", default="Qwen3-1.7B")
  parser.add_argument("--prompt", default="What is 2+2?")
  parser.add_argument(
      "--vary-prompts-per-process",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "If True (default), each JAX process uses a DIFFERENT prompt"
          " selected from a built-in list keyed by jax.process_index() %"
          " len(prompts). This reproduces the multi-host data pattern that"
          " GRPO uses (each process tokenizes its own GSM8K question) and"
          " is the test that distinguishes a chat-template bug from a"
          " sampler/sharding multi-host data-routing bug. When False, the"
          " script uses --prompt on every process (original behavior)."
      ),
  )
  parser.add_argument(
      "--use-gsm8k-system-prompt",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "Apply the same GSM8K system prompt that math_dataset.apply_template"
          " uses, so the chat-templated string matches GRPO's prompt format"
          " exactly."
      ),
  )
  parser.add_argument("--repeat-prompts", type=int, default=1)
  parser.add_argument("--max-new-tokens", type=int, default=32)
  parser.add_argument("--max-prompt-length", type=int, default=256)
  parser.add_argument("--cache-size", type=int, default=512)
  parser.add_argument("--top-k-logits", type=int, default=20)
  parser.add_argument("--temperature", type=float, default=0.0)
  parser.add_argument("--top-p", type=float, default=None)
  parser.add_argument("--top-k", type=int, default=None)
  parser.add_argument(
      "--mesh-shape",
      default=None,
      help=(
          "Comma-separated mesh shape, e.g. 8,4. Defaults to 8,4 for"
          " multihost TPU and 1,<device_count> otherwise."
      ),
  )
  parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument(
      "--model-download-path",
      default=None,
      help=(
          "Local directory for Tunix's HF download. Defaults to"
          " /tmp/tunix_debug_models/<model-name>."
      ),
  )
  return parser.parse_args()


def _mesh_shape_from_arg(mesh_shape_arg: str | None) -> tuple[int, int]:
  if mesh_shape_arg:
    parts = tuple(int(x.strip()) for x in mesh_shape_arg.split(","))
    if len(parts) != 2:
      raise ValueError(f"--mesh-shape must have two dimensions: {mesh_shape_arg}")
    return parts
  if jax.process_count() > 1 and jax.device_count() == 32:
    return (8, 4)
  return (1, jax.device_count())


def _make_mesh(mesh_shape_arg: str | None) -> jax.sharding.Mesh:
  mesh_shape = _mesh_shape_from_arg(mesh_shape_arg)
  devices = np.asarray(jax.devices(), dtype=object).reshape(mesh_shape)
  return jax.sharding.Mesh(devices, ("fsdp", "tp"))


def _format_token(tokenizer: tokenizer_adapter.TokenizerAdapter, token_id: int) -> str:
  try:
    return repr(tokenizer.decode([int(token_id)], skip_special_tokens=False))
  except TypeError:
    return repr(tokenizer.decode([int(token_id)]))


def _print_top_logits(
    logits: np.ndarray,
    tokenizer: tokenizer_adapter.TokenizerAdapter,
    top_k: int,
) -> None:
  logits = np.asarray(logits, dtype=np.float32)
  top_ids = np.argsort(logits)[-top_k:][::-1]
  print(f"\nTOP_{top_k}_FIRST_COMPLETION_LOGITS:")
  for rank, token_id in enumerate(top_ids, start=1):
    print(
        f"{rank:02d} id={int(token_id):6d} "
        f"logit={float(logits[token_id]): .6f} "
        f"text={_format_token(tokenizer, int(token_id))}"
    )


def _tree_first_leaf_dtype(tree: object) -> str:
  leaves = jax.tree.leaves(tree)
  if not leaves:
    return "<no leaves>"
  return str(getattr(leaves[0], "dtype", type(leaves[0])))


def main() -> None:
  args = _parse_args()
  _maybe_initialize_jax_distributed()
  model_id = f"Qwen/{args.model_name}"
  hf_token = os.environ.get("HF_TOKEN")
  model_download_path = args.model_download_path or os.path.join(
      "/tmp", "tunix_debug_models", args.model_name
  )

  print("ENV:")
  print(f"  hostname={socket.gethostname()}")
  print(f"  JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS')}")
  print(f"  TUNIX_JAX_DISTRIBUTED_AUTO_INIT={os.environ.get('TUNIX_JAX_DISTRIBUTED_AUTO_INIT')}")
  print(f"  jax.default_backend()={jax.default_backend()}")
  print(f"  jax.process_index()={jax.process_index()}")
  print(f"  jax.process_count()={jax.process_count()}")
  print(f"  jax.device_count()={jax.device_count()}")
  print(f"  jax.local_device_count()={jax.local_device_count()}")
  print(f"  jax.devices()={jax.devices()}")
  print(f"  model_id={model_id}")
  print(f"  model_download_path={model_download_path}")

  if args.vary_prompts_per_process and jax.process_count() > 1:
    chosen_prompt = _PER_PROCESS_PROMPTS[
        jax.process_index() % len(_PER_PROCESS_PROMPTS)
    ]
  else:
    chosen_prompt = args.prompt

  print(
      f"\nPER_PROCESS_PROMPT_SELECTION: process_index={jax.process_index()}"
      f" vary_prompts_per_process={args.vary_prompts_per_process}"
      f" chosen_prompt={chosen_prompt!r}"
  )

  hf_tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
  if args.use_gsm8k_system_prompt:
    # Mirror tunix/examples/data/math_dataset.py SYSTEM_PROMPT exactly so the
    # chat-templated string matches the one GRPO feeds the rollout.
    gsm8k_system_prompt = (
        "You are given a problem. Think about the problem and provide your"
        " reasoning. Place it between <think> and </think>. Then, provide"
        " the final answer (i.e., just one numerical value) between <answer>"
        " and </answer>."
    )
    messages = [
        {"role": "system", "content": gsm8k_system_prompt},
        {"role": "user", "content": chosen_prompt},
    ]
  else:
    messages = [{"role": "user", "content": chosen_prompt}]

  chat_text = hf_tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True,
      enable_thinking=args.enable_thinking,
  )
  hf_prompt_ids = hf_tokenizer(chat_text, add_special_tokens=False).input_ids

  print("\nCHAT_TEMPLATE_TEXT_REPR:")
  print(repr(chat_text))
  print("\nHF_PROMPT_TOKEN_IDS:")
  print(hf_prompt_ids)
  print("\nHF_PROMPT_DECODED:")
  print(hf_tokenizer.decode(hf_prompt_ids, skip_special_tokens=False))

  tokenizer = tokenizer_adapter.Tokenizer(
      tokenizer_type="huggingface",
      tokenizer_path=model_id,
      add_bos=False,
      add_eos=False,
      hf_access_token=hf_token,
  )
  tunix_prompt_ids = tokenizer.encode(chat_text, add_special_tokens=False)
  print("\nTUNIX_TOKENIZER_PROMPT_TOKEN_IDS:")
  print(tunix_prompt_ids)
  print(f"\nTOKENIZER_IDS_MATCH={list(hf_prompt_ids) == list(tunix_prompt_ids)}")

  mesh = _make_mesh(args.mesh_shape)
  print(f"mesh.shape={mesh.shape}")
  print(f"mesh.axis_names={mesh.axis_names}")
  print("\nLoading Tunix model...")
  model, resolved_model_path = automodel.AutoModel.from_pretrained(
      model_id=model_id,
      mesh=mesh,
      model_source=automodel.ModelSource.HUGGINGFACE,
      model_download_path=model_download_path,
  )
  print(f"resolved_model_path={resolved_model_path}")
  print(f"model_config_dtype={getattr(model.config, 'dtype', None)}")
  print(f"model_config_param_dtype={getattr(model.config, 'param_dtype', None)}")
  print(f"state_first_leaf_dtype={_tree_first_leaf_dtype(nnx.state(model))}")
  if not hasattr(model, "num_embed") and hasattr(model.config, "vocab_size"):
    # Tunix Sampler's optional logits buffer expects this legacy attribute.
    model.num_embed = model.config.vocab_size

  rollout_sampler = tunix_sampler.Sampler(
      model,
      tokenizer,
      tunix_sampler.CacheConfig(
          cache_size=args.cache_size,
          num_layers=model.config.num_layers,
          num_kv_heads=model.config.num_kv_heads,
          head_dim=model.config.head_dim,
      ),
  )

  top_p = args.top_p
  top_k = args.top_k
  temperature = args.temperature
  if temperature == 0.0 and top_p is not None:
    raise ValueError("Use either greedy temperature=0.0 or top_p sampling, not both.")

  print("\nRunning Tunix sampler...")
  # IMPORTANT: enter the mesh context so the Sampler sees a non-empty
  # `pxla.thread_resources.env.physical_mesh`. Without this, shard_input
  # silently no-ops and the JIT'd prefill/decode runs on local arrays only,
  # which is not what GRPO actually does (rl_cluster.generate enters the
  # rollout mesh context before calling the sampler).
  with mesh:
    output = rollout_sampler(
        input_strings=[chat_text] * args.repeat_prompts,
        max_generation_steps=args.max_new_tokens,
        max_prompt_length=args.max_prompt_length,
        echo=False,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=0,
        pad_output=True,
        return_logits=True,
        eos_tokens=[tokenizer.eos_id()],
    )

  tokens = np.asarray(output.tokens[0])
  logits = np.asarray(output.logits[0])
  nonpad = tokens[tokens != tokenizer.pad_id()]

  print(
      f"\nPER_PROCESS_RESULT_HEADER: process_index={jax.process_index()}"
      f" chosen_prompt={chosen_prompt!r}"
  )
  print("\nTUNIX_COMPLETION_TOKEN_IDS_PADDED:")
  print(tokens.tolist())
  print("\nTUNIX_COMPLETION_TOKEN_IDS_NONPAD:")
  print(nonpad.tolist())
  print("\nTUNIX_COMPLETION_DECODED:")
  print(tokenizer.decode(nonpad.tolist(), skip_special_tokens=False))
  print("\nTUNIX_FULL_TEXT_FROM_SAMPLER:")
  print(output.text[0])
  if len(output.text) > 1:
    print("\nTUNIX_ALL_COMPLETIONS:")
    for i, text in enumerate(output.text):
      print(f"[{i}] {text!r}")

  print(
      "\nPER_PROCESS_RESULT_SUMMARY:"
      f" process_index={jax.process_index()}"
      f" prompt_first_60={chosen_prompt[:60]!r}"
      f" completion_first_120={output.text[0][:120]!r}"
  )

  if logits.size:
    _print_top_logits(logits[0], tokenizer, args.top_k_logits)


if __name__ == "__main__":
  main()
