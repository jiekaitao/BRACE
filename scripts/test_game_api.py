#!/usr/bin/env python3
"""End-to-end test script for the basketball game analysis API.

Usage:
    python scripts/test_game_api.py path/to/basketball.mp4 [--host localhost:8001]

Uploads a video, submits it for game analysis, polls progress,
and prints the final player results.
"""

import argparse
import json
import sys
import time

import requests


def main():
    parser = argparse.ArgumentParser(description="Test basketball game analysis API")
    parser.add_argument("video", help="Path to basketball MP4 file")
    parser.add_argument("--host", default="localhost:8001", help="Backend host:port")
    args = parser.parse_args()

    base = f"http://{args.host}"

    # Step 1: Health check
    print("Checking backend health...")
    try:
        r = requests.get(f"{base}/health", timeout=5)
        r.raise_for_status()
        print(f"  OK: {r.json()}")
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  Is the backend running? Try: docker compose up -d")
        sys.exit(1)

    # Step 2: Upload video
    print(f"\nUploading {args.video}...")
    with open(args.video, "rb") as f:
        r = requests.post(f"{base}/api/upload", files={"file": f}, timeout=120)
    r.raise_for_status()
    upload = r.json()
    session_id = upload["session_id"]
    print(f"  Uploaded: session_id={session_id}")

    # Step 3: Submit for game analysis
    print("\nSubmitting for basketball game analysis...")
    r = requests.post(f"{base}/api/games", params={"session_id": session_id}, timeout=30)
    if r.status_code != 200:
        print(f"  FAILED ({r.status_code}): {r.text}")
        sys.exit(1)
    game = r.json()
    game_id = game["game_id"]
    print(f"  Game created: game_id={game_id}")

    # Step 4: Poll progress
    print("\nProcessing...")
    last_progress = -1
    while True:
        time.sleep(3)
        r = requests.get(f"{base}/api/games/{game_id}", timeout=10)
        r.raise_for_status()
        status = r.json()

        progress = status.get("progress", 0)
        pct = int(progress * 100)
        if pct != last_progress:
            player_count = status.get("player_count", 0)
            print(f"  {pct}% — frame {status.get('frame_idx', 0)}/{status.get('total_frames', '?')} — {player_count} players detected")
            last_progress = pct

        if status["status"] == "complete":
            print("\n  Game analysis complete!")
            break
        elif status["status"] == "failed":
            print(f"\n  FAILED: {status.get('error', 'unknown error')}")
            sys.exit(1)

    # Step 5: Print game summary
    print(f"\nGame Summary:")
    print(f"  Duration: {status.get('duration_sec', 0):.1f}s")
    print(f"  Players: {status.get('player_count', 0)}")
    if status.get("team_colors"):
        print(f"  Teams:")
        for tc in status["team_colors"]:
            print(f"    Team {tc['team_id']}: {tc['color_name']} (RGB {tc['rgb']})")

    # Step 6: Fetch and print player details
    print(f"\nPlayers:")
    r = requests.get(f"{base}/api/games/{game_id}/players", timeout=10)
    r.raise_for_status()
    players = r.json()["players"]

    for p in players:
        jersey = p.get("jersey_number") or "?"
        team = p.get("team_id")
        color = p.get("jersey_color") or "unknown"
        sid = p.get("subject_id")

        print(f"\n  Player #{jersey} (subject {sid}, team {team}, {color})")

        summary = p.get("analysis_summary") or {}
        print(f"    Frames: {summary.get('total_frames', '?')}, "
              f"Segments: {summary.get('n_segments', '?')}, "
              f"Clusters: {summary.get('n_clusters', '?')}")

        injuries = p.get("injury_events", [])
        if injuries:
            print(f"    Injury events: {len(injuries)}")
            for ie in injuries[:5]:
                print(f"      - {ie}")
        else:
            print(f"    No injury events")

        quality = p.get("final_quality")
        if quality:
            form = quality.get("form_score")
            if form is not None:
                print(f"    Form score: {form}")

    # Step 7: Fetch timeline for first player
    if players:
        first = players[0]
        sid = first["subject_id"]
        print(f"\nTimeline for player {sid}:")
        r = requests.get(f"{base}/api/games/{game_id}/players/{sid}", timeout=10)
        r.raise_for_status()
        detail = r.json()
        timeline = detail.get("timeline", [])
        print(f"  {len(timeline)} snapshots")
        for snap in timeline[:5]:
            print(f"    Frame {snap['frame_idx']}: {json.dumps(snap.get('biomechanics') or {}, default=str)[:100]}")

    print("\nDone!")


if __name__ == "__main__":
    main()
