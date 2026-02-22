#!/usr/bin/env bash
# Download sample basketball videos for testing.
# Videos are saved to data/sports_videos/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data/sports_videos"

mkdir -p "$DATA_DIR"

echo "=== Basketball Test Video Downloader ==="
echo "Target directory: $DATA_DIR"
echo ""

# Check if yt-dlp is available
if ! command -v yt-dlp &> /dev/null; then
    echo "yt-dlp not found. Install with: pip install yt-dlp"
    echo "Or download a sample video manually to: $DATA_DIR/basketball.mp4"
    exit 1
fi

echo "Downloading sample basketball footage..."
echo "NOTE: You can also manually place any basketball MP4 in $DATA_DIR/basketball.mp4"
echo ""

# Download a short basketball clip (public domain / Creative Commons)
# This is a placeholder URL — replace with an actual public basketball video URL
if [ ! -f "$DATA_DIR/basketball.mp4" ]; then
    echo "No basketball.mp4 found."
    echo "Please manually download a basketball video and save it as:"
    echo "  $DATA_DIR/basketball.mp4"
    echo ""
    echo "Suggested sources:"
    echo "  - Pexels.com (free stock video)"
    echo "  - Pixabay.com (free stock video)"
    echo "  - YouTube (with yt-dlp, ensure you have rights)"
else
    echo "basketball.mp4 already exists, skipping download."
fi

echo ""
echo "Done. Available videos:"
ls -lh "$DATA_DIR"/*.mp4 2>/dev/null || echo "  (none found)"
