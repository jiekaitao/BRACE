#!/usr/bin/env bash
# Download CC0-licensed demo exercise videos from Pexels for testing.
# These are direct CDN links to free (Pexels license) videos.

set -euo pipefail

DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/videos"
mkdir -p "$DEST_DIR"

echo "Downloading demo videos to $DEST_DIR ..."

# Squats video (~15 reps, good for testing repeated motion clustering)
if [ ! -f "$DEST_DIR/squats.mp4" ]; then
  echo "  → squats.mp4"
  curl -fsSL -o "$DEST_DIR/squats.mp4" \
    "https://videos.pexels.com/video-files/4761437/4761437-uhd_2560_1440_25fps.mp4"
  echo "    ✓ squats.mp4 downloaded"
else
  echo "  → squats.mp4 (already exists, skipping)"
fi

# Jumping jacks video (high-velocity, tests segmentation)
if [ ! -f "$DEST_DIR/jumping_jacks.mp4" ]; then
  echo "  → jumping_jacks.mp4"
  curl -fsSL -o "$DEST_DIR/jumping_jacks.mp4" \
    "https://videos.pexels.com/video-files/4761444/4761444-uhd_2560_1440_25fps.mp4"
  echo "    ✓ jumping_jacks.mp4 downloaded"
else
  echo "  → jumping_jacks.mp4 (already exists, skipping)"
fi

# Push-ups / lunges video (different movement pattern)
if [ ! -f "$DEST_DIR/pushups.mp4" ]; then
  echo "  → pushups.mp4"
  curl -fsSL -o "$DEST_DIR/pushups.mp4" \
    "https://videos.pexels.com/video-files/4761449/4761449-uhd_2560_1440_25fps.mp4"
  echo "    ✓ pushups.mp4 downloaded"
else
  echo "  → pushups.mp4 (already exists, skipping)"
fi

echo ""
echo "Done! Demo videos saved to $DEST_DIR"
echo "  - squats.mp4"
echo "  - jumping_jacks.mp4"
echo "  - pushups.mp4"
