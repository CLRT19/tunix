#!/bin/bash
set -e

BUCKET_NAME="${1:-linrong-vlm-tpu-us-central1-a}"
MOUNT_POINT="${2:-$HOME/bucket}"

# ── Helpers ──────────────────────────────────────────────────────────────────

error() { echo "ERROR: $*" >&2; exit 1; }

require() {
    command -v "$1" &>/dev/null || error "'$1' is not installed. Install it with: $2"
}

bucket_location() {
    local bucket_name="$1"
    local output
    local attempt
    local location

    if command -v gcloud &>/dev/null; then
        for attempt in 1 2 3; do
            if output=$(gcloud storage buckets describe "gs://${bucket_name}" \
                --format="value(location)" 2>&1); then
                location=$(echo "$output" | awk '
                    NF == 1 {
                        candidate=tolower($1)
                        if (candidate ~ /^(us|eu|asia)$/ || candidate ~ /^[a-z]+(-[a-z0-9]+)+$/) {
                            loc=candidate
                        }
                    }
                    END {
                        if (loc) {
                            print loc
                        } else {
                            exit 1
                        }
                    }'
                ) && {
                    echo "$location"
                    return 0
                }
                echo "gcloud bucket location output did not contain a location: ${output}" >&2
                return 1
            fi
            echo "gcloud bucket location query failed on attempt ${attempt}: ${output}" >&2
            sleep "$attempt"
        done
    fi

    if command -v gsutil &>/dev/null; then
        for attempt in 1 2 3; do
            if output=$(gsutil ls -L -b "gs://${bucket_name}" 2>&1); then
                location=$(echo "$output" | awk '/Location constraint:/ {loc=tolower($NF)} END {if (loc) print loc; else exit 1}') && {
                    echo "$location"
                    return 0
                }
                echo "gsutil bucket location output did not contain a location: ${output}" >&2
                return 1
            fi
            echo "gsutil bucket location query failed on attempt ${attempt}: ${output}" >&2
            sleep "$attempt"
        done
    fi

    return 1
}

# ── Prereqs ───────────────────────────────────────────────────────────────────

require curl   "sudo apt-get install -y curl"
if ! command -v gcloud &>/dev/null && ! command -v gsutil &>/dev/null; then
    error "'gcloud' or 'gsutil' is required. Install the Google Cloud SDK."
fi

if ! command -v gcsfuse &>/dev/null; then
    echo "gcsfuse not found — installing via Google Cloud apt repo..."
    GCSFUSE_REPO="gcsfuse-$(lsb_release -c -s)"
    echo "deb https://packages.cloud.google.com/apt ${GCSFUSE_REPO} main" \
        | sudo tee /etc/apt/sources.list.d/gcsfuse.list
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | sudo apt-key add -
    sudo apt-get update -q
    sudo apt-get install -y gcsfuse
fi

# ── Get VM zone from GCE metadata server ─────────────────────────────────────

echo "Detecting VM zone..."
ZONE_FULL=$(curl -sf \
    "http://metadata.google.internal/computeMetadata/v1/instance/zone" \
    -H "Metadata-Flavor: Google") \
    || error "Could not reach GCE metadata server. Are you running on a GCP VM?"

# Zone comes back as projects/PROJECT_ID/zones/us-central1-a — strip the prefix
VM_ZONE="${ZONE_FULL##*/}"
VM_REGION="${VM_ZONE%-*}"   # us-central1-a → us-central1

echo "  VM zone:   $VM_ZONE"
echo "  VM region: $VM_REGION"

# ── Get bucket region ─────────────────────────────────────────────────────────

echo "Detecting bucket region for gs://${BUCKET_NAME}..."
BUCKET_LOCATION=$(bucket_location "$BUCKET_NAME") \
    || error "Could not query bucket 'gs://${BUCKET_NAME}'. Check the name and your permissions."

[ -z "$BUCKET_LOCATION" ] && error "Could not determine location for bucket '${BUCKET_NAME}'."

echo "  Bucket region: $BUCKET_LOCATION"

# ── Region check ──────────────────────────────────────────────────────────────

if [ "$BUCKET_LOCATION" != "$VM_REGION" ]; then
    error "Region mismatch — bucket is in '$BUCKET_LOCATION' but VM is in '$VM_REGION'.
       Cross-region transfers cost \$0.02–0.05/GB (see TPU_GUIDE.md § Cost).
       Either use a bucket in '$VM_REGION' or move your VM to '$BUCKET_LOCATION'."
fi

echo "Region check passed: both VM and bucket are in '$VM_REGION'."

# ── Mount ─────────────────────────────────────────────────────────────────────

mkdir -p "$MOUNT_POINT"

if mountpoint -q "$MOUNT_POINT"; then
    echo "Bucket already mounted at $MOUNT_POINT — nothing to do."
    exit 0
fi

echo "Mounting gs://${BUCKET_NAME} at ${MOUNT_POINT}..."
gcsfuse --implicit-dirs "$BUCKET_NAME" "$MOUNT_POINT"

echo "Done. Access your bucket at: $MOUNT_POINT"
echo "To unmount: fusermount -u $MOUNT_POINT"
