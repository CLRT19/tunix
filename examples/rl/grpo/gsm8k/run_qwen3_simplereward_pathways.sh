# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Multi-host Pathways launcher for run_qwen3_simplereward.sh.
# Submits the existing single-VM script as an xpk Pathways workload. See
# docs/quickstart.md "Quick Start: Multi-Node Training" for the cluster /
# image prerequisites this assumes are already in place.

set -euo pipefail
set -x

# --- xpk / cluster knobs (defaults match scripts/setup_pathways_cluster.sh) ---
PROJECT=${PROJECT:-vision-mix}
ZONE=${ZONE:-us-central1-a}
REGION=${REGION:-${ZONE%-*}}
CLUSTER_NAME=${CLUSTER_NAME:-tunix-${USER}-pw}
TPU_TYPE=${TPU_TYPE:-v5p-16}
WORKLOAD_NAME=${WORKLOAD_NAME:-tunix-${USER}-qwen3-simplereward-$(date +%Y%m%d-%H%M%S)}
NUM_SLICES=${NUM_SLICES:-1}
PRIORITY=${PRIORITY:-medium}
BASE_DOCKER_IMAGE=${BASE_DOCKER_IMAGE:-${REGION}-docker.pkg.dev/${PROJECT}/tunix-images/tunix_base_image:latest}

# xpk shells out to gcloud and needs zone/project to be discoverable. Set them
# in gcloud config so subsequent gcloud/kubectl calls don't prompt.
gcloud config set project "${PROJECT}"
gcloud config set compute/zone "${ZONE}"

# Path to the inner training script INSIDE the docker image. The repo Dockerfile
# uses WORKDIR /app and `COPY . .`, so files keep their repo-relative layout.
INNER_SCRIPT=${INNER_SCRIPT:-/app/examples/rl/grpo/gsm8k/run_qwen3_simplereward.sh}
# Run from the directory holding base_config.yaml so the relative path used by
# the inner script resolves correctly.
INNER_CWD=${INNER_CWD:-/app/tunix/cli}

# --- Pass-through training knobs (defaults match run_qwen3_simplereward.sh) ---
model_name=${model_name:-Qwen3-1.7B-base}
batch_size=${batch_size:-1}
num_batches=${num_batches:-3738}
num_train_epochs=${num_train_epochs:-1}
warmup_ratio=${warmup_ratio:-0.1}
train_fraction=${train_fraction:-1.0}

# --- Build the workload command ----------------------------------------------
# JAX_PLATFORMS=proxy + JAX_BACKEND_TARGET point JAX at the Pathways proxy
# colocated with the head pod; grpo_main.py auto-initializes pathwaysutils when
# it sees JAX_PLATFORMS=proxy.
COMMAND="cd ${INNER_CWD} && \
TPU_MIN_LOG_LEVEL=0 TF_CPP_MIN_LOG_LEVEL=0 TPU_STDERR_LOG_LEVEL=0 \
JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 \
ENABLE_PATHWAYS_PERSISTENCE=1 \
model_name=${model_name} batch_size=${batch_size} num_batches=${num_batches} \
num_train_epochs=${num_train_epochs} warmup_ratio=${warmup_ratio} \
train_fraction=${train_fraction} \
bash ${INNER_SCRIPT}"

xpk workload create-pathways \
  --cluster="${CLUSTER_NAME}" \
  --workload="${WORKLOAD_NAME}" \
  --command="${COMMAND}" \
  --num-slices="${NUM_SLICES}" \
  --tpu-type="${TPU_TYPE}" \
  --zone="${ZONE}" \
  --project="${PROJECT}" \
  --base-docker-image "${BASE_DOCKER_IMAGE}" \
  --priority="${PRIORITY}"
