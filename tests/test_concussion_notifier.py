import asyncio
import json
from pathlib import Path

from backend.concussion_pipeline import CoachReportNotifier


def test_notifier_persists_to_sent_when_no_delivery_channel(tmp_path: Path):
    async def run_test():
        notifier = CoachReportNotifier(
            outbox_dir=tmp_path,
            webhook_url="",
            smtp_host="",
            default_coach_email="",
            retry_seconds=0.01,
        )
        await notifier.queue_report(
            report={"play_id": "p1", "player_id": "u2", "risk_level": "HIGH"},
            coach_email="coach@example.com",
        )
        await notifier.flush_once()
        assert not any((tmp_path / "pending").glob("*.json"))
        sent_files = list((tmp_path / "sent").glob("*.json"))
        assert len(sent_files) == 1
        payload = json.loads(sent_files[0].read_text(encoding="utf-8"))
        assert payload["coach_email"] == "coach@example.com"

    asyncio.run(run_test())


def test_notifier_retries_when_webhook_fails(tmp_path: Path):
    async def run_test():
        notifier = CoachReportNotifier(
            outbox_dir=tmp_path,
            webhook_url="http://localhost:9999/does-not-exist",
            smtp_host="",
            retry_seconds=0.01,
        )
        await notifier.queue_report(
            report={"play_id": "p2", "player_id": "u4", "risk_level": "MODERATE"},
            coach_email=None,
        )

        pending_files = list((tmp_path / "pending").glob("*.json"))
        assert len(pending_files) == 1
        payload = json.loads(pending_files[0].read_text(encoding="utf-8"))
        assert payload["attempts"] >= 1
        assert float(payload["next_retry_ts"]) > 0

    asyncio.run(run_test())
