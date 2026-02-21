#!/bin/bash
# Download multi-person sports clips for testing multi-subject tracking.
# Requires: brew install yt-dlp

set -euo pipefail

OUTDIR="data/sports_videos"
mkdir -p "$OUTDIR"

FORMAT="bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]"

echo "=== Downloading multi-person sports clips ==="
echo "Output directory: $OUTDIR"
echo ""

# Basketball — pickup game / practice drills (multiple players on court)
echo "[1/4] Basketball..."
yt-dlp --format "$FORMAT" \
  --download-sections "*0:10-1:00" \
  --force-keyframes-at-cuts \
  -o "$OUTDIR/basketball.mp4" \
  --no-playlist \
  "https://www.youtube.com/watch?v=5WvjGMXZfpg" \
  2>/dev/null || echo "  Skipped (video unavailable or yt-dlp error)"

# Boxing — sparring match (2 fighters)
echo "[2/4] Boxing..."
yt-dlp --format "$FORMAT" \
  --download-sections "*0:30-1:30" \
  --force-keyframes-at-cuts \
  -o "$OUTDIR/boxing.mp4" \
  --no-playlist \
  "https://www.youtube.com/watch?v=VNynTdqFkHw" \
  2>/dev/null || echo "  Skipped (video unavailable or yt-dlp error)"

# Tennis — doubles or singles with visible line judges
echo "[3/4] Tennis..."
yt-dlp --format "$FORMAT" \
  --download-sections "*0:15-1:00" \
  --force-keyframes-at-cuts \
  -o "$OUTDIR/tennis.mp4" \
  --no-playlist \
  "https://www.youtube.com/watch?v=Kq-VejC3MZs" \
  2>/dev/null || echo "  Skipped (video unavailable or yt-dlp error)"

# Soccer — training session with multiple players
echo "[4/4] Soccer..."
yt-dlp --format "$FORMAT" \
  --download-sections "*0:10-1:00" \
  --force-keyframes-at-cuts \
  -o "$OUTDIR/soccer.mp4" \
  --no-playlist \
  "https://www.youtube.com/watch?v=2Rga0DP2Vnc" \
  2>/dev/null || echo "  Skipped (video unavailable or yt-dlp error)"

echo ""
echo "=== Done ==="
ls -lh "$OUTDIR/"
