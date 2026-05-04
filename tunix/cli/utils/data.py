"""Utilities for handling and loading datasets in tunix CLI."""

import ast
import functools
import importlib
import os
from typing import Any, Callable, Optional, Union

from absl import logging
import jax
from tunix.generate import tokenizer_adapter

Tokenizer = tokenizer_adapter.Tokenizer
TokenizerAdapter = tokenizer_adapter.TokenizerAdapter


def apply_chat_template(x, tokenizer: TokenizerAdapter) -> dict[str, Any]:
  return {
      "prompts": tokenizer.apply_chat_template(
          x["prompt"], tokenize=False, add_generation_prompt=True
      ),
      **{k: v for k, v in x.items() if k != "prompt"},
  }


def parse_call_string(arg_string: str) -> tuple[list[Any], dict[str, Any]]:
  """Parses a string containing function call arguments and keyword arguments.

  Args:
    arg_string: A string representing the arguments of a function call,
      e.g., "'arg1', 123, kwarg1='value', kwarg2=456".

  Returns:
    A tuple containing two elements:
      - A list of positional arguments.
      - A dictionary of keyword arguments.

  Raises:
    ValueError: If the arg_string is not a valid argument syntax.
  """
  if not arg_string.strip():
    return [], {}

  fake_expression = f"dummy_func({arg_string})"
  try:
    tree = ast.parse(fake_expression)
  except SyntaxError as exc:
    raise ValueError(f"Invalid argument syntax: {arg_string}") from exc

  if not tree.body or not isinstance(tree.body[0], ast.Expr):
    raise ValueError(
        f"Internal error: Expected an expression node for '{arg_string}'"
    )

  call_node = tree.body[0].value
  if not isinstance(call_node, ast.Call):
    raise ValueError(f"Internal error: Expected a Call node for '{arg_string}'")

  parsed_args = []
  for node in call_node.args:
    parsed_args.append(ast.literal_eval(node))

  parsed_kwargs = {}
  for keyword in call_node.keywords:
    parsed_kwargs[keyword.arg] = ast.literal_eval(keyword.value)

  return parsed_args, parsed_kwargs


def get_dataset_from_module(specifier: str, tokenizer: TokenizerAdapter):
  """Get dataset from module.

  Examples of specifier:
    - "data.coding" # create_dataset is the default function
    - "data.coding:create_dataset"
    - "data.coding:get_my_dataset"
    - "data.coding:create_dataset(name='coding_v0')"
    - "data.coding:create_dataset('coding_v0', split='train')"
    - "/home/user/project/data/coding.py:get_dataset"

  Args:
    specifier: The specifier of the module.
    tokenizer: The tokenizer to apply to the dataset.

  Returns:
    The dataset.
  Raises:
    ImportError: If the module cannot be imported or loaded.
  """
  if "(" in specifier and ":" in specifier:
    specifier, args_part = specifier.rsplit("(", 1)
  else:
    args_part = ""
  if ":" in specifier:
    specifier, func_spec = specifier.rsplit(":", 1)
  else:
    func_spec = ""
  if os.path.exists(specifier) and specifier.endswith(".py"):
    module_name = os.path.splitext(os.path.basename(specifier))[0]
    spec = importlib.util.spec_from_file_location(module_name, specifier)
    module = importlib.util.module_from_spec(spec)

    if spec is None:
      raise ImportError(f"Failed to create spec for {specifier}")
    if spec.loader is None:
      raise ImportError(f"Failed to get loader for spec {specifier}")
    if module is None:
      raise ImportError(f"Failed to create module for {specifier}")

    try:
      spec.loader.exec_module(module)
    except Exception as e:
      raise ImportError(
          f"Failed to execute module {module_name} from {specifier}: {e}"
      ) from e
  else:
    try:
      module = importlib.import_module(specifier)
    except Exception as e:
      raise ImportError(f"Failed to import module {specifier}: {e}") from e
  args = []
  kwargs = {}
  if func_spec:
    func = getattr(module, func_spec)
    if args_part:
      args_part = args_part.rstrip(")")
      args, kwargs = parse_call_string(args_part)

  else:
    func = module.create_dataset
  dataset = func(*args, **kwargs)
  return dataset.map(
      functools.partial(apply_chat_template, tokenizer=tokenizer)
  )


def post_init_dataset(
    dataset,
    tokenizer: TokenizerAdapter,
    batch_size: int,
    num_batches: Optional[int],
    max_prompt_length: Optional[int],
    fraction: float = 1.0,
    num_epochs: int = 1,
    prompt_key: str = "prompts",
    custom_batch_fn: Optional[Callable] = None,
    shard_by_process: bool = False,
    batch_size_is_global: bool = False,
):
  """Applies post-initialization transformations to a dataset.

  This function filters, batches, and optionally limits the number of batches
  in a dataset.

  Args:
    dataset: The input dataset.
    tokenizer: The tokenizer used for prompt length filtering.
    batch_size: The size of each batch.
    num_batches: If not None, the maximum number of batches to yield.
    max_prompt_length: If not None and greater than 0, prompts longer than this
      will be filtered out.
    fraction: Fraction of the dataset to use (between 0.0 and 1.0), commonly
      used for splitting training and validation sets.
    num_epochs: Number of times to repeat the dataset.
    shard_by_process: Whether to give each JAX process a disjoint shard of the
      dataset before batching.
    batch_size_is_global: Whether `batch_size` is the global batch size across
      all JAX processes. When true, each process batches
      `batch_size // jax.process_count()` examples.
  Returns:
    The processed dataset.
  """
  process_count = jax.process_count() if shard_by_process else 1
  process_index = jax.process_index() if shard_by_process else 0
  local_batch_size = batch_size
  if batch_size_is_global and process_count > 1:
    if batch_size % process_count != 0:
      raise ValueError(
          "Global batch_size must be divisible by jax.process_count() when"
          f" sharding the dataset. Got batch_size={batch_size},"
          f" process_count={process_count}."
      )
    local_batch_size = batch_size // process_count

  if max_prompt_length is not None and max_prompt_length > 0:

    def prompt_length_filter(x):
      tokens = tokenizer.encode(x[prompt_key])
      return len(tokens) <= max_prompt_length

    dataset = dataset.filter(prompt_length_filter)

  if num_batches is not None:
    target_size = min(num_batches * batch_size, len(dataset))
    dataset = dataset[:target_size]

  if fraction < 1.0 and fraction > 0.0:
    first_segment_size = int(len(dataset) * fraction)
    first_segment_dataset = dataset[:first_segment_size]
    second_segment_dataset = dataset[first_segment_size:]
  else:
    first_segment_dataset = dataset
    second_segment_dataset = None

  def process_shard(dataset_segment):
    if not shard_by_process:
      return dataset_segment
    segment_size = len(dataset_segment)
    shardable_size = segment_size - (segment_size % process_count)
    local_size = shardable_size // process_count
    if local_size == 0:
      raise ValueError(
          "Dataset segment is too small to shard across JAX processes. Got"
          f" segment_size={segment_size}, process_count={process_count}."
      )
    local_start = process_index * local_size
    local_end = local_start + local_size
    if shardable_size != segment_size:
      logging.warning(
          "Dropping %d dataset examples so %d examples can be evenly sharded"
          " across %d JAX processes.",
          segment_size - shardable_size,
          shardable_size,
          process_count,
      )
    return dataset_segment[local_start:local_end]

  first_segment_dataset = process_shard(first_segment_dataset)
  if second_segment_dataset is not None:
    second_segment_dataset = process_shard(second_segment_dataset)

  if shard_by_process:
    logging.info(
        "Prepared process-local dataset shard: process=%d/%d,"
        " global_batch_size=%d, local_batch_size=%d, local_examples=%d.",
        process_index,
        process_count,
        batch_size,
        local_batch_size,
        len(first_segment_dataset),
    )

  first_segment_dataset = (
      first_segment_dataset.repeat(num_epochs)
      .to_iter_dataset()
      .batch(local_batch_size, batch_fn=custom_batch_fn)
  )
  if second_segment_dataset is not None:
    second_segment_dataset = (
        second_segment_dataset.repeat(num_epochs)
        .to_iter_dataset()
        .batch(local_batch_size, batch_fn=custom_batch_fn)
    )

  return first_segment_dataset, second_segment_dataset
