"""Tests for host-only barriers around vLLM's local initialization."""

import pickle

import numpy as np

from tunix.generate import base_sampler
from tunix.rl.rollout import base_rollout
from tunix.rl.rollout import vllm_rollout


def test_vllm_phase_barriers_are_noop_on_one_process(monkeypatch):
  monkeypatch.setattr(vllm_rollout.jax, "process_count", lambda: 1)
  vllm_rollout._wait_for_all_actor_weights()
  vllm_rollout._wait_for_all_vllm_engines()
  vllm_rollout._wait_for_all_vllm_weight_loads()


def test_vllm_phase_barriers_use_coordination_service(monkeypatch):
  calls = []

  class FakeClient:
    def wait_at_barrier(self, name, timeout_ms):
      calls.append((name, timeout_ms))

  monkeypatch.setattr(vllm_rollout.jax, "process_count", lambda: 16)
  monkeypatch.setattr(vllm_rollout.jax, "process_index", lambda: 3)
  monkeypatch.setattr(
      vllm_rollout.jax_distributed.global_state, "client", FakeClient()
  )
  monkeypatch.setenv("QWEN3VL_RUN_ID", "barrier-test")
  monkeypatch.setenv("QWEN3VL_MODEL_LOAD_BARRIER_TIMEOUT_MS", "1234")

  vllm_rollout._wait_for_all_actor_weights()
  vllm_rollout._wait_for_all_vllm_engines()
  vllm_rollout._wait_for_all_vllm_weight_loads()

  assert calls == [
      ("barrier-test_vllm_actor_weights_ready_before_engine_init", 1234),
      ("barrier-test_vllm_engine_ready_before_weight_load", 1234),
      ("barrier-test_vllm_local_weights_ready_before_training", 1234),
  ]


def test_vllm_sampler_initializes_in_process_order(monkeypatch):
  events = []

  class FakeClient:
    def wait_at_barrier(self, name, timeout_ms):
      events.append(("barrier", name, timeout_ms))

  monkeypatch.setattr(vllm_rollout.jax, "process_count", lambda: 4)
  monkeypatch.setattr(vllm_rollout.jax, "process_index", lambda: 2)
  monkeypatch.setattr(
      vllm_rollout.jax_distributed.global_state, "client", FakeClient()
  )
  monkeypatch.setenv("QWEN3VL_RUN_ID", "wave-test")
  monkeypatch.setenv("QWEN3VL_VLLM_INIT_WAVE_SIZE", "1")
  monkeypatch.setenv("QWEN3VL_MODEL_LOAD_BARRIER_TIMEOUT_MS", "4321")
  monkeypatch.delenv("QWEN3VL_VLLM_GLOBAL_MESH", raising=False)
  sampler = object()

  def factory():
    events.append(("factory", 2))
    return sampler

  assert vllm_rollout._initialize_vllm_sampler_in_waves(factory) is sampler
  assert events == [
      ("barrier", "wave-test_vllm_init_wave_0_1", 4321),
      ("barrier", "wave-test_vllm_init_wave_1_2", 4321),
      ("factory", 2),
      ("barrier", "wave-test_vllm_init_wave_2_3", 4321),
      ("barrier", "wave-test_vllm_init_wave_3_4", 4321),
  ]


def test_global_vllm_sampler_initializes_on_every_process(monkeypatch):
  events = []
  sampler = object()
  monkeypatch.setattr(vllm_rollout.jax, "process_count", lambda: 8)
  monkeypatch.setenv("QWEN3VL_VLLM_GLOBAL_MESH", "1")
  monkeypatch.setenv("QWEN3VL_VLLM_INIT_WAVE_SIZE", "1")

  def factory():
    events.append("factory")
    return sampler

  assert vllm_rollout._initialize_vllm_sampler_in_waves(factory) is sampler
  assert events == ["factory"]


def test_global_vllm_gathers_host_ordered_prompts_and_images(monkeypatch):
  host_payloads = [
      pickle.dumps(
          ([f"p{host}-0", f"p{host}-1"], [f"i{host}-0", f"i{host}-1"]),
          protocol=pickle.HIGHEST_PROTOCOL,
      )
      for host in range(3)
  ]
  lengths = np.asarray([len(payload) for payload in host_payloads])
  padded_length = int(lengths.max())
  padded_payloads = np.zeros((3, padded_length), dtype=np.uint8)
  for host, payload in enumerate(host_payloads):
    padded_payloads[host, : len(payload)] = np.frombuffer(
        payload, dtype=np.uint8
    )
  calls = 0

  def fake_allgather(value, *, tiled):
    nonlocal calls
    assert tiled is True
    calls += 1
    if calls == 1:
      return np.asarray([2, 2, 2], dtype=np.int32)
    if calls == 2:
      return lengths
    if calls == 3:
      return padded_payloads.reshape(-1)
    raise AssertionError(f"Unexpected allgather call {calls}: {value.shape}")

  monkeypatch.setattr(vllm_rollout.jax, "process_count", lambda: 3)
  monkeypatch.setattr(vllm_rollout.jax, "process_index", lambda: 1)
  monkeypatch.setattr(
      vllm_rollout.multihost_utils, "process_allgather", fake_allgather
  )

  prompts, images, local_slice = (
      vllm_rollout._gather_global_rollout_inputs(
          ["p1-0", "p1-1"], ["i1-0", "i1-1"]
      )
  )

  assert prompts == [
      "p0-0", "p0-1", "p1-0", "p1-1", "p2-0", "p2-1"
  ]
  assert images == [
      "i0-0", "i0-1", "i1-0", "i1-1", "i2-0", "i2-1"
  ]
  assert local_slice == slice(2, 4)


def test_global_vllm_returns_only_the_callers_local_rows(monkeypatch):
  class FakeSampler:
    last_terminated = [[False, False, True, False, False, True]]

    def __call__(self, **kwargs):
      self.kwargs = kwargs
      return base_sampler.SamplerOutput(
          text=[f"t{i}" for i in range(6)],
          logits=None,
          tokens=[np.asarray([i], dtype=np.int32) for i in range(6)],
          padded_prompt_tokens=np.arange(12, dtype=np.int32).reshape(6, 2),
          logprobs=[[float(i)] for i in range(6)],
      )

  monkeypatch.setenv("QWEN3VL_VLLM_GLOBAL_MESH", "1")
  monkeypatch.setattr(vllm_rollout.jax, "process_count", lambda: 3)
  monkeypatch.setattr(
      vllm_rollout,
      "_gather_global_rollout_inputs",
      lambda prompts, images: (
          [f"p{i}" for i in range(6)],
          [f"i{i}" for i in range(6)],
          slice(2, 4),
      ),
  )
  rollout = vllm_rollout.VllmRollout.__new__(vllm_rollout.VllmRollout)
  rollout._sampler = FakeSampler()

  output = rollout.generate(
      ["p2", "p3"],
      base_rollout.RolloutConfig(max_tokens_to_generate=4),
      images=["i2", "i3"],
  )

  assert rollout._sampler.kwargs["input_strings"] == [
      "p0", "p1", "p2", "p3", "p4", "p5"
  ]
  assert rollout._sampler.kwargs["images"] == [
      "i0", "i1", "i2", "i3", "i4", "i5"
  ]
  assert output.text == ["t2", "t3"]
  assert [tokens.tolist() for tokens in output.tokens] == [[2], [3]]
  assert output.left_padded_prompt_tokens.tolist() == [[4, 5], [6, 7]]
  assert output.logprobs == [[2.0], [3.0]]
  assert output.terminated == [True, False]
