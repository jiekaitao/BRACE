#!/bin/bash
# Clone the kooksung pathological gait dataset into data/
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"

if [ -d "$DATA_DIR/pathological_gait_datasets" ]; then
    echo "Dataset already exists at $DATA_DIR/pathological_gait_datasets"
    exit 0
fi

echo "Cloning pathological gait dataset..."
cd "$DATA_DIR"
git clone https://github.com/kooksung/pathological_gait_datasets.git
echo "Done. Dataset cloned to $DATA_DIR/pathological_gait_datasets"
