#!/usr/bin/env python3
r"""Full-pipeline test: YOLO + IdentityResolver + SceneDetector + Gemini + PlayerRiskEngine.

Usage:
    python scripts/test_risk_engine_live.py "C:\Users\Fabio Jorge\Downloads\Basketball Video #1.mp4"
    python scripts/test_risk_engine_live.py "path/to/video.mp4" --seconds 30
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Load .env before anything else (Gemini API key)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# Add backend to path (same as Docker flat layout)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import cv2
import numpy as np


# Basketball-specific Gemini prompt (same as basketball_processor.py)
_BASKETBALL_CLASSIFY_PROMPT = (
    "What basketball activity is this player performing? "
    "Respond with ONLY a single word or short phrase. "
    "Prefer these specific labels when applicable: "
    "dunk, dunking, rebound, rebounding, block, blocking, alley-oop, slam dunk, "
    "crossover, euro step, cutting, direction change, "
    "shooting, jump shot, free throw, layup, three pointer, floater, hook shot, "
    "dribbling, ball handling, "
    "defensive slide, guarding, closeout, defensive shuffle, "
    "running, sprinting, jogging, walking, "
    "jump, landing. "
    "Other valid labels: passing, screening, boxing out, fast break, "
    "catching, standing, stretching."
)


def main():
    parser = argparse.ArgumentParser(description="Full-pipeline PlayerRiskEngine test")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--seconds", type=int, default=60, help="Seconds to process (default: 60)")
    parser.add_argument("--model", default="yolo11x-pose.pt", help="YOLO model name")
    parser.add_argument("--no-gemini", action="store_true", help="Disable Gemini classification")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    # ---- ANSI colors ----
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    DIM = "\033[2m"
    BLINK = "\033[5m"
    STATUS_COLOR = {"GREEN": GREEN, "YELLOW": YELLOW, "RED": RED}

    # ---- Load pipeline components ----
    print(f"{BOLD}Loading pipeline components...{RESET}")

    from legacy_backend import LegacyPoseBackend
    from subject_manager import SubjectManager
    from player_risk_engine import PlayerRiskEngine
    from identity_resolver import IdentityResolver
    from embedding_extractor import DummyExtractor
    from scene_detector import InlineSceneDetector

    # YOLO pose backend
    print(f"  YOLO model: {args.model} ...", end="", flush=True)
    backend = LegacyPoseBackend(model_name=args.model, conf_threshold=0.3)
    backend.warmup()
    print(f" {GREEN}OK{RESET} ({backend.model.device})")

    # Identity resolver (DummyExtractor — no OSNet on Windows, but still handles
    # track-to-subject mapping and gap tolerance)
    print(f"  IdentityResolver (DummyExtractor) ...", end="", flush=True)
    reid_extractor = DummyExtractor()
    resolver = IdentityResolver(reid_extractor)
    print(f" {GREEN}OK{RESET}")

    # Scene detector
    print(f"  SceneDetector ...", end="", flush=True)
    scene_detector = InlineSceneDetector()
    print(f" {GREEN}OK{RESET}")

    # Gemini activity classification
    gemini = None
    if not args.no_gemini:
        try:
            from gemini_classifier import GeminiActivityClassifier
            gemini = GeminiActivityClassifier()
            if gemini.available:
                print(f"  Gemini 2.0 Flash ... {GREEN}OK{RESET} (activity classification enabled)")
            else:
                print(f"  Gemini ... {YELLOW}unavailable{RESET} (no API key)")
                gemini = None
        except Exception as e:
            print(f"  Gemini ... {YELLOW}skipped{RESET} ({e})")
    else:
        print(f"  Gemini ... {DIM}disabled{RESET}")

    print()

    # ---- Open video ----
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(args.seconds * fps)
    duration = total_frames / fps

    print(f"{BOLD}Video:{RESET} {width}x{height} @ {fps:.1f}fps, {duration:.0f}s total")
    print(f"{BOLD}Processing:{RESET} first {args.seconds}s ({max_frames} frames)")
    print(f"{'=' * 70}")
    print()

    # ---- Initialize analysis components ----
    manager = SubjectManager(fps=fps, cluster_threshold=2.0)
    risk_engine = PlayerRiskEngine(fps=fps)
    executor = ThreadPoolExecutor(max_workers=4)

    # Gemini frame buffer (ring buffer for representative frame extraction)
    frame_buffer: dict[int, np.ndarray] = {}
    _FRAME_BUFFER_MAX = 600

    def _get_buffered_frame(idx: int) -> np.ndarray | None:
        return frame_buffer.get(idx)

    def _classify_cluster_bg(analyzer, cluster_id: int, gem):
        """Background Gemini classification for a cluster."""
        indices = analyzer.get_cluster_frame_indices(cluster_id)
        if not indices:
            analyzer.set_activity_label(cluster_id, "unknown")
            return
        bbox = analyzer.get_cluster_bbox(cluster_id)
        if bbox is None:
            analyzer.set_activity_label(cluster_id, "unknown")
            return
        frames = gem.get_representative_frames(indices, _get_buffered_frame, count=4)
        if not frames:
            analyzer.set_activity_label(cluster_id, "unknown")
            return
        label = gem.classify_activity(frames, bbox, prompt=_BASKETBALL_CLASSIFY_PROMPT)
        analyzer.set_activity_label(cluster_id, label)

    # Track scene cuts
    scene_cut_count = 0

    # ---- Process frames ----
    frame_idx = 0
    t_start = time.monotonic()
    last_print_time = -1.0

    try:
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Store in ring buffer for Gemini
            frame_buffer[frame_idx] = rgb
            if len(frame_buffer) > _FRAME_BUFFER_MAX:
                oldest = min(frame_buffer)
                del frame_buffer[oldest]

            # Scene cut detection
            if scene_detector.process_frame(rgb):
                backend.on_scene_cut()
                resolver.on_scene_cut()
                scene_cut_count += 1

            # Pipeline: detect + track
            results = backend.process_frame(rgb)

            # Identity resolution
            resolved = None
            if results:
                resolved = resolver.resolve_pipeline_results(results, rgb, width, height)

            # Process each person through analyzers
            items = resolved if resolved is not None else results
            for item in items:
                if resolved is not None:
                    rp = item
                    pr = rp.pipeline_result
                    subject_id = rp.subject_id
                else:
                    pr = item
                    subject_id = pr.track_id

                landmarks_xyzv = pr.landmarks_mp
                analyzer = manager.get_or_create_analyzer(subject_id)
                analyzer.last_seen_frame = frame_idx

                response = analyzer.process_frame(landmarks_xyzv, img_wh=(width, height))

                # Re-analysis + Gemini classification
                if analyzer.needs_reanalysis():
                    analyzer.run_analysis()

                    if gemini is not None and gemini.available:
                        for cid in analyzer.get_clusters_needing_classification():
                            analyzer.mark_classification_pending(cid)
                            executor.submit(_classify_cluster_bg, analyzer, cid, gemini)

                # Feed into risk engine
                quality = response.get("quality")
                cluster_quality = None
                if hasattr(analyzer, '_quality_tracker') and analyzer._quality_tracker is not None:
                    qt = analyzer._quality_tracker
                    cid = response.get("cluster_id")
                    if cid is not None and hasattr(qt, '_cluster_state'):
                        cs = qt._cluster_state.get(cid)
                        if cs is not None:
                            cluster_quality = cs.latest_quality
                activity_name = None
                if hasattr(analyzer, '_quality_tracker') and analyzer._quality_tracker is not None:
                    profile = analyzer._quality_tracker._current_profile
                    if profile is not None:
                        activity_name = profile.name

                risk_engine.process_frame(
                    subject_id=subject_id,
                    frame_idx=frame_idx,
                    video_time=frame_idx / fps,
                    quality=quality,
                    cluster_quality=cluster_quality,
                    activity_profile_name=activity_name,
                )

            # Cleanup stale subjects and resolver tracks
            manager.cleanup_stale(frame_idx)
            active_track_ids = {pr.track_id for pr in results}
            resolver.cleanup_stale_tracks(active_track_ids)

            # Print status every second
            video_time = frame_idx / fps
            if video_time - last_print_time >= 1.0:
                last_print_time = video_time
                elapsed = time.monotonic() - t_start
                proc_fps = (frame_idx + 1) / max(elapsed, 0.001)
                active_ids = manager.get_active_track_ids()

                # Progress bar
                progress_pct = (frame_idx + 1) / max_frames * 100
                bar_len = 30
                filled = int(bar_len * progress_pct / 100)
                bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
                sys.stdout.write(
                    f"{DIM}[{bar}] {progress_pct:5.1f}% | "
                    f"t={video_time:5.1f}s | {proc_fps:5.1f} fps | "
                    f"{len(active_ids)} active | "
                    f"cuts={scene_cut_count}{RESET}\n"
                )

                # Per-player status (only active players, sorted by ID)
                if active_ids:
                    for sid in sorted(active_ids):
                        state = risk_engine._get_state(sid)
                        status = state.status.value
                        color = STATUS_COLOR.get(status, RESET)
                        label = manager.get_label(sid)

                        parts = []
                        if state.fatigue_window:
                            parts.append(f"fatigue={state.fatigue_window[-1]:.2f}")
                        if state.form_window:
                            parts.append(f"form={state.form_window[-1]:.0f}")

                        active_events = [e for e in state.injury_events if e.active]
                        if active_events:
                            parts.append(f"injuries={len(active_events)}")

                        # Show activity label if set
                        if hasattr(analyzer, '_quality_tracker') and analyzer._quality_tracker is not None:
                            p = analyzer._quality_tracker._current_profile
                            if p and p.name != "generic":
                                parts.append(f"{CYAN}{p.display_name}{RESET}")

                        pull_str = ""
                        if state.pull_recommended:
                            pull_str = f" {BLINK}{RED}\u25cf PULL{RESET}"

                        detail = " | ".join(parts) if parts else "warming up"

                        sys.stdout.write(
                            f"  {BOLD}{label:>8}{RESET} "
                            f"[{color}{BOLD}{status:>6}{RESET}] "
                            f"{detail}{pull_str}\n"
                        )
                else:
                    sys.stdout.write(f"  {DIM}(no players detected){RESET}\n")

                sys.stdout.flush()

            frame_idx += 1

    except KeyboardInterrupt:
        print(f"\n{YELLOW}Interrupted at frame {frame_idx}{RESET}")

    cap.release()
    executor.shutdown(wait=False)
    elapsed = time.monotonic() - t_start

    # ---- Collect confirmed subjects from resolver ----
    confirmed = resolver.get_confirmed_subjects(min_frames=int(fps))  # 1 second
    confirmed_sids = set(confirmed.keys()) if confirmed else None

    # ---- Final summary ----
    print()
    print(f"{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}FINAL SUMMARY{RESET}")
    print(f"  Processed {frame_idx} frames in {elapsed:.1f}s ({frame_idx/max(elapsed,0.001):.1f} fps)")
    print(f"  Scene cuts detected: {scene_cut_count}")
    total_subjects = len(risk_engine._states)
    confirmed_count = len(confirmed_sids) if confirmed_sids else total_subjects
    print(f"  Total track IDs: {total_subjects}, Confirmed players (>{1:.0f}s): {confirmed_count}")
    print(f"{'=' * 70}")

    # Only show confirmed players (or all if no resolver)
    show_sids = sorted(confirmed_sids) if confirmed_sids else sorted(risk_engine._states.keys())

    # Separate into risk categories for display
    red_players = []
    yellow_players = []
    green_players = []

    for sid in show_sids:
        summary = risk_engine.get_player_summary(sid)
        summary["_sid"] = sid
        summary["_label"] = manager.get_label(sid)
        status = summary["risk_status"]
        if status == "RED":
            red_players.append(summary)
        elif status == "YELLOW":
            yellow_players.append(summary)
        else:
            green_players.append(summary)

    def _print_player(summary):
        sid = summary["_sid"]
        label = summary["_label"]
        status = summary["risk_status"]
        color = STATUS_COLOR.get(status, RESET)
        wl = summary["workload"]
        events = summary["injury_events"]
        active_events = [e for e in events if e["active"]]
        closed_events = [e for e in events if not e["active"]]

        time_str = f"{wl['total_frames']/fps:.1f}s visible"
        effort_str = f"{wl['high_effort_pct']:.0f}% high-effort" if wl['high_effort_frames'] > 0 else ""

        print(f"\n  {BOLD}{label}{RESET}  {color}{BOLD}[{status}]{RESET}  "
              f"{DIM}{time_str}{f', {effort_str}' if effort_str else ''}{RESET}")

        if wl["activity_distribution"]:
            dist = ", ".join(
                f"{k}: {v}" for k, v in
                sorted(wl["activity_distribution"].items(), key=lambda x: -x[1])[:3]
            )
            print(f"    Activities: {dist}")

        if events:
            print(f"    Injury events: {BOLD}{len(active_events)} active{RESET}, {len(closed_events)} closed")
            for e in events[:5]:
                ev_color = RED if e["severity"] == "high" else YELLOW
                tag = f"{BOLD}ACTIVE{RESET}" if e["active"] else f"{DIM}closed{RESET}"
                print(f"      {ev_color}[{e['severity'].upper()}]{RESET} "
                      f"{e['joint']} {e['risk_type']} "
                      f"@ {e['onset_time']:.1f}s ({e['duration_sec']:.1f}s) [{tag}]")
            if len(events) > 5:
                print(f"      {DIM}... and {len(events)-5} more{RESET}")

        if summary["pull_recommended"]:
            print(f"    {RED}{BOLD}\u25cf PULL RECOMMENDED:{RESET} {', '.join(summary['pull_reasons'])}")

        history = summary["risk_history"]
        if history:
            counts: dict[str, int] = {}
            for h in history:
                counts[h["status"]] = counts.get(h["status"], 0) + 1
            total_h = len(history)
            pcts = "  ".join(
                f"{STATUS_COLOR.get(s, '')}{s}: {c/total_h*100:.0f}%{RESET}"
                for s, c in sorted(counts.items(), key=lambda x: ["RED","YELLOW","GREEN"].index(x[0]) if x[0] in ["RED","YELLOW","GREEN"] else 99)
            )
            print(f"    Timeline: {pcts}")

    # Print RED first, then YELLOW, then GREEN
    if red_players:
        print(f"\n{RED}{BOLD}--- HIGH RISK ---{RESET}")
        for p in red_players:
            _print_player(p)

    if yellow_players:
        print(f"\n{YELLOW}{BOLD}--- CAUTION ---{RESET}")
        for p in yellow_players:
            _print_player(p)

    if green_players:
        print(f"\n{GREEN}{BOLD}--- LOW RISK ---{RESET}")
        # Compact display for green players
        for p in green_players:
            sid = p["_sid"]
            label = p["_label"]
            wl = p["workload"]
            events = p["injury_events"]
            time_str = f"{wl['total_frames']/fps:.1f}s"
            ev_str = f", {len(events)} injury events" if events else ""
            print(f"  {label:>8}  {GREEN}[GREEN]{RESET}  {time_str}{ev_str}")

    # Overall stats
    print(f"\n{'=' * 70}")
    print(f"{BOLD}Risk distribution:{RESET}  "
          f"{RED}{len(red_players)} RED{RESET}  "
          f"{YELLOW}{len(yellow_players)} YELLOW{RESET}  "
          f"{GREEN}{len(green_players)} GREEN{RESET}  "
          f"(of {len(show_sids)} confirmed players)")
    if gemini:
        stats = gemini.get_stats()
        print(f"{BOLD}Gemini usage:{RESET}  "
              f"{stats.get('api_calls', 0)} API calls, "
              f"{stats.get('cache_hits', 0)} cache hits, "
              f"~${stats.get('estimated_cost_usd', 0):.4f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
