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
# Pathways smoke-test launcher. Submits run_qwen3_simplereward_smoke.sh as
# an xpk Pathways workload so the multi-host setup can be debugged with
# minute-scale iterations. Reuses the same image / cluster / env wiring as
# run_qwen3_simplereward_pathways.sh.
#
# Usage:
#   CLUSTER_NAME=my-cluster TPU_TYPE=v5p-16 \
#     bash examples/rl/grpo/gsm8k/run_qwen3_simplereward_pathways_smoke.sh
#
# Tail logs after submit:
#   xpk workload list --cluster=$CLUSTER_NAME
#   kubectl logs -f <jobset-pod-name> -c jax-tpu        # head
#   kubectl logs -f <jobset-pod-name> -c pathways-worker # worker

set -euo pipefail
set -x

# Defaults match what scripts/setup_pathways_cluster.sh provisions.
PROJECT=${PROJECT:-vision-mix}
ZONE=${ZONE:-us-central1-a}
REGION=${REGION:-${ZONE%-*}}
CLUSTER_NAME=${CLUSTER_NAME:-tunix-${USER}-pw}
TPU_TYPE=${TPU_TYPE:-v5p-16}
WORKLOAD_NAME=${WORKLOAD_NAME:-tunix-${USER}-smoke-$(date +%Y%m%d-%H%M%S)}
NUM_SLICES=${NUM_SLICES:-1}
PRIORITY=${PRIORITY:-medium}
BASE_DOCKER_IMAGE=${BASE_DOCKER_IMAGE:-${REGION}-docker.pkg.dev/${PROJECT}/tunix-images/tunix_base_image:latest}

# xpk shells out to gcloud and needs zone/project to be discoverable.
gcloud config set project "${PROJECT}"
gcloud config set compute/zone "${ZONE}"

INNER_SCRIPT=${INNER_SCRIPT:-/app/examples/rl/grpo/gsm8k/run_qwen3_simplereward_smoke.sh}
INNER_CWD=${INNER_CWD:-/app/tunix/cli}

# Smoke knobs forwarded to the inner script. Override any to widen the test.
SMOKE_NUM_BATCHES=${SMOKE_NUM_BATCHES:-2}
SMOKE_TOTAL_GEN_STEPS=${SMOKE_TOTAL_GEN_STEPS:-64}
SMOKE_MAX_PROMPT_LEN=${SMOKE_MAX_PROMPT_LEN:-128}
SMOKE_MESH_SHAPE=${SMOKE_MESH_SHAPE:-"(2,4)"}
SMOKE_NUM_GENERATIONS=${SMOKE_NUM_GENERATIONS:-2}

COMMAND="cd ${INNER_CWD} && \
TPU_MIN_LOG_LEVEL=0 TF_CPP_MIN_LOG_LEVEL=0 TPU_STDERR_LOG_LEVEL=0 \
JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 \
ENABLE_PATHWAYS_PERSISTENCE=1 \
num_batches=${SMOKE_NUM_BATCHES} \
total_generation_steps=${SMOKE_TOTAL_GEN_STEPS} \
max_prompt_length=${SMOKE_MAX_PROMPT_LEN} \
mesh_shape='${SMOKE_MESH_SHAPE}' \
num_generations=${SMOKE_NUM_GENERATIONS} \
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

set +x
echo
echo "Submitted smoke workload: ${WORKLOAD_NAME}"
echo "Watch:  xpk workload list --cluster=${CLUSTER_NAME}"
echo "Logs:   kubectl logs -f \$(kubectl get pod -l xpk.google.com/workload=${WORKLOAD_NAME} -o name | head -n1) -c jax-tpu"
echo "Cancel: xpk workload delete --cluster=${CLUSTER_NAME} --workload=${WORKLOAD_NAME}"
