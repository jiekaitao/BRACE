#!/bin/bash
# Download 16+ multi-person sports videos for pipeline testing.
# Each clip is 15-30 seconds, 480p, targeting scenes with 2+ visible people.
set -euo pipefail

OUTDIR="data/sports_videos"
mkdir -p "$OUTDIR"

YT="yt-dlp --js-runtimes node --no-playlist --no-warnings"
FMT="bestvideo[height<=480][ext=mp4]+bestaudio/best[height<=480]"

download() {
    local name="$1" url="$2" start="$3" end="$4"
    local out="$OUTDIR/${name}.mp4"
    if [ -f "$out" ]; then
        echo "  [skip] $name already exists"
        return 0
    fi
    echo "  [dl] $name ..."
    $YT --format "$FMT" \
        --download-sections "*${start}-${end}" \
        --force-keyframes-at-cuts \
        -o "$out" "$url" 2>/dev/null && echo "  [ok] $name" || echo "  [FAIL] $name"
}

echo "=== Downloading multi-person sports videos ==="

# --- Basketball ---
download "basketball_pickup"    "https://www.youtube.com/watch?v=mhZVmMT6YwI" "0:30" "1:00"
download "basketball_drill"     "https://www.youtube.com/watch?v=Nw2G9OaqJrU" "0:15" "0:45"

# --- Soccer / Football ---
download "soccer_training"      "https://www.youtube.com/watch?v=SHqx3s7Twzg" "0:20" "0:50"
download "soccer_match"         "https://www.youtube.com/watch?v=8YkbeXpGZbk" "0:30" "1:00"

# --- Boxing / MMA ---
download "boxing_sparring"      "https://www.youtube.com/watch?v=gJKOXZ_kgHo" "0:10" "0:40"
download "mma_training"         "https://www.youtube.com/watch?v=hxQ6tLhMoqQ" "0:15" "0:45"

# --- Tennis ---
download "tennis_doubles"       "https://www.youtube.com/watch?v=SqBoWKnG6kk" "0:20" "0:50"
download "tennis_rally"         "https://www.youtube.com/watch?v=V0Y1HRG4JNE" "0:10" "0:35"

# --- Swimming ---
download "swimming_race"        "https://www.youtube.com/watch?v=2MX30tREqmo" "0:10" "0:40"

# --- Group fitness / CrossFit ---
download "crossfit_class"       "https://www.youtube.com/watch?v=mlsSpQpNnfI" "0:15" "0:45"
download "group_workout"        "https://www.youtube.com/watch?v=gC_L9qAHVJ8" "0:10" "0:40"

# --- Track and field ---
download "sprint_race"          "https://www.youtube.com/watch?v=3nbjhpcZ9_g" "0:05" "0:25"
download "relay_race"           "https://www.youtube.com/watch?v=4VdXF0bHRxo" "0:10" "0:35"

# --- Volleyball ---
download "volleyball_match"     "https://www.youtube.com/watch?v=t4mBlnDF6wQ" "0:20" "0:50"

# --- Wrestling / Martial Arts ---
download "wrestling_match"      "https://www.youtube.com/watch?v=MeL1kSfHG3o" "0:10" "0:40"

# --- Rugby ---
download "rugby_training"       "https://www.youtube.com/watch?v=F3i3Hslm5rs" "0:15" "0:45"

# --- Dance / Cheerleading ---
download "dance_group"          "https://www.youtube.com/watch?v=4hrAx_TW1eg" "0:20" "0:45"

# --- Gymnastics ---
download "gymnastics_floor"     "https://www.youtube.com/watch?v=VqfP7K8J0aI" "0:05" "0:30"

echo ""
echo "=== Download summary ==="
ls -lhS "$OUTDIR/"*.mp4 2>/dev/null || echo "No files downloaded"
echo ""
echo "Total files: $(ls "$OUTDIR/"*.mp4 2>/dev/null | wc -l | tr -d ' ')"
