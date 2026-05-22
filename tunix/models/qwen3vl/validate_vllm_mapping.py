"""Step-1 validation for the Qwen3-VL tunix->vLLM weight mapping.

Runs in the tunix env (jax 0.10.1), NO TPU needed. Checks:
  A. Every real tunix nnx param (from an abstract eval_shape model) is covered by
     exactly one mapping source pattern  -> catches silent-skip (unmapped source).
  B. Every mapping source pattern matches >=1 real param                -> no dead entries.
  C. Mapping targets, expanded, are an exact bijection with the tpu-inference
     load_weights() target inventory (replicated below as the fixture)   -> no orphan
     vLLM params (unsynced) and no wrong target names.
  D. The patch-embed hook turns the real tunix [C*T*P*P, hidden] kernel into the
     tpu-inference Conv3D kernel shape [T, P, P, C, hidden].
"""
import os
import re
import sys

import jax.numpy as jnp
from flax import nnx

from tunix.models.qwen3vl import BACKEND_MAPPINGS
from tunix.models.qwen3vl import model as M

VJ = BACKEND_MAPPINGS["vllm_jax"]
MAPPING = VJ["to_hf_mappings"]
HOOKS = VJ["to_hf_hook_fns"] or {}

# Validate against either model size; 8B is untied (has lm_head), 4B is tied.
_SIZE = os.environ.get("QWEN3VL_VALIDATE_SIZE", "4b").lower()
cfg = getattr(M.ModelConfig, f"qwen3vl_{_SIZE}")()
cfg.remat_config = M.RematConfig.NONE
vc = cfg.vision_config
n_layers = cfg.num_layers
depth = vc.depth
n_deep = len(vc.deepstack_visual_indexes)
tie = getattr(cfg, "use_tied_embedding", False)
print(f"[cfg] {_SIZE}: text_layers={n_layers} vision_depth={depth} "
      f"deepstack={n_deep} tie_embed={tie} "
      f"patch(C={vc.in_channels},T={vc.temporal_patch_size},P={vc.patch_size}) "
      f"vhidden={vc.hidden_size}")


def pat_to_re(p):
  return re.compile("^" + re.escape(p).replace(r"\*", r"\d+") + "$")


# ---- build the REAL tunix source param tree (abstract, no allocation) ----
abs_model = nnx.eval_shape(lambda: M.Qwen3VL(cfg, rngs=nnx.Rngs(params=0)))
_, state = nnx.split(abs_model)
src = {}
for keys, leaf in state.flat_state():
  k = ".".join(str(x) for x in keys)
  if "rng" in k:
    continue
  val = getattr(leaf, "value", leaf)
  src[k] = getattr(val, "shape", None)
print(f"[src] real tunix params (non-rng leaves): {len(src)}")

src_pats = [(s, pat_to_re(s), tgt) for s, (tgt, _) in MAPPING.items()]

# ---- A + B: source coverage / dead patterns ----
unmapped, multi = [], []
used_pat = set()
src_to_tgt_concrete = {}
for k in sorted(src):
  hits = [(s, tgt) for s, rx, tgt in src_pats if rx.match(k)]
  if not hits:
    unmapped.append(k)
  elif len(hits) > 1:
    multi.append((k, [h[0] for h in hits]))
  else:
    s, tgtpat = hits[0]
    used_pat.add(s)
    # concretise target by substituting the layer index captured from k
    m = re.match("^" + re.escape(s).replace(r"\*", r"(\d+)") + "$", k)
    idx = m.groups()
    t = tgtpat
    for g in idx:
      t = t.replace("*", g, 1)
    src_to_tgt_concrete[k] = t
dead = [s for s, _, _ in src_pats if s not in used_pat]
# lm_head.w is intentionally absent when embeddings are tied (4B); the entry is
# kept for the untied 8B. A dead lm_head.w under tie=True is expected, not a bug.
if tie:
  dead = [s for s in dead if s != "lm_head.w"]

# ---- C: expected tpu-inference target inventory (replicated from load_weights) ----
expected = set()
for i in range(n_layers):
  p = f"language_model.layers.{i}"
  expected |= {
      f"{p}.input_layernorm.weight",
      f"{p}.post_attention_layernorm.weight",
      f"{p}.self_attn.q_proj.weight",
      f"{p}.self_attn.k_proj.weight",
      f"{p}.self_attn.v_proj.weight",
      f"{p}.self_attn.o_proj.weight",
      f"{p}.self_attn.q_norm.weight",
      f"{p}.self_attn.k_norm.weight",
      f"{p}.mlp.gate_proj.weight",
      f"{p}.mlp.up_proj.weight",
      f"{p}.mlp.down_proj.weight",
  }
expected |= {"language_model.embed_tokens.weight", "language_model.norm.weight"}
if not tie:
  expected.add("lm_head.weight")
for i in range(depth):
  p = f"visual.blocks.{i}"
  expected |= {
      f"{p}.attn.qkv_proj.kernel", f"{p}.attn.qkv_proj.bias",
      f"{p}.attn.proj.kernel", f"{p}.attn.proj.bias",
      f"{p}.mlp.fc1.kernel", f"{p}.mlp.fc1.bias",
      f"{p}.mlp.fc2.kernel", f"{p}.mlp.fc2.bias",
      f"{p}.norm1.scale", f"{p}.norm1.bias",
      f"{p}.norm2.scale", f"{p}.norm2.bias",
  }
expected |= {
    "visual.patch_embed.proj.kernel", "visual.patch_embed.proj.bias",
    "visual.pos_embed.embedding",
    "visual.merger.norm.scale", "visual.merger.norm.bias",
    "visual.merger.linear_fc1.kernel", "visual.merger.linear_fc1.bias",
    "visual.merger.linear_fc2.kernel", "visual.merger.linear_fc2.bias",
}
for j in range(n_deep):
  p = f"visual.deepstack_merger_list.{j}"
  expected |= {
      f"{p}.norm.scale", f"{p}.norm.bias",
      f"{p}.linear_fc1.kernel", f"{p}.linear_fc1.bias",
      f"{p}.linear_fc2.kernel", f"{p}.linear_fc2.bias",
  }

produced = set(src_to_tgt_concrete.values())
orphan_targets = sorted(expected - produced)   # vLLM params NOT written by sync
extra_targets = sorted(produced - expected)    # mapping targets not in vLLM model

# ---- D: patch-embed hook shape ----
hook_ok, hook_msg = True, ""
pe = "visual.patch_embed.proj.kernel"
if pe in src and pe in HOOKS:
  flat, hidden = src[pe]
  out = HOOKS[pe](jnp.zeros((flat, hidden), jnp.float32))
  want = (vc.temporal_patch_size, vc.patch_size, vc.patch_size, vc.in_channels, hidden)
  hook_ok = tuple(out.shape) == want
  hook_msg = f"{(flat, hidden)} -> {tuple(out.shape)} (want {want})"
else:
  hook_ok, hook_msg = False, f"patch_embed src/hook missing (src={pe in src}, hook={pe in HOOKS})"

# ---------------- report ----------------
print("\n=== A. unmapped real source params (FATAL: silent-skip) ===")
print("  none" if not unmapped else "\n".join("  " + k for k in unmapped))
print("=== multi-match source params (FATAL) ===")
print("  none" if not multi else "\n".join(f"  {k}: {p}" for k, p in multi))
print("=== B. dead mapping source patterns (match no real param) ===")
print("  none" if not dead else "\n".join("  " + s for s in dead))
print(f"\n=== C. target bijection vs tpu-inference load_weights "
      f"(expected={len(expected)}, produced={len(produced)}) ===")
print("  orphan targets (vLLM param NOT synced):")
print("    none" if not orphan_targets else "\n".join("    " + k for k in orphan_targets))
print("  extra targets (mapping target not in vLLM model):")
print("    none" if not extra_targets else "\n".join("    " + k for k in extra_targets))
print(f"\n=== D. patch-embed hook shape ===\n  {'OK' if hook_ok else 'FAIL'}: {hook_msg}")

ok = (not unmapped and not multi and not dead and not orphan_targets
      and not extra_targets and hook_ok)
print("\n==================  " + ("ALL CHECKS PASSED" if ok else "FAILURES PRESENT")
      + "  ==================")
sys.exit(0 if ok else 1)
