"""Small all-worker JAX TPU initialization probe for TPU VM launch scripts."""

import os
import socket
import time

import jax


def main() -> None:
  start = time.time()
  print(
      socket.gethostname(),
      "backend",
      jax.default_backend(),
      "process",
      jax.process_index(),
      "/",
      jax.process_count(),
      "devices",
      jax.device_count(),
      "local",
      jax.local_device_count(),
      "elapsed",
      round(time.time() - start, 2),
      flush=True,
  )
  time.sleep(int(os.environ.get("TPU_PREFLIGHT_HOLD_SECONDS", "60")))


if __name__ == "__main__":
  main()
