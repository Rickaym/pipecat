#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Any

import pytest

from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
from pipecat.services.speechmatics.stt import SpeechmaticsSTTService


def _segment(text: str, **overrides: Any) -> dict[str, Any]:
    segment: dict[str, Any] = {
        "text": text,
        "speaker_id": "S1",
        "is_active": True,
        "language": "en",
        "timestamp": "2026-01-01T00:00:00.000Z",
        "results": [{"alternatives": [{"content": text}]}],
        "is_eou": False,
    }
    segment.update(overrides)
    return segment


class _RecordingService(SpeechmaticsSTTService):
    """Captures pushed frames instead of running a real pipeline."""

    def __init__(self):
        super().__init__(api_key="test-key")
        self.pushed: list[Any] = []

    async def push_frame(self, frame, direction=None):
        self.pushed.append(frame)

    async def _handle_transcription(self, *args, **kwargs):
        return None

    def finalized_texts(self) -> list[str]:
        return [f.text for f in self.pushed if isinstance(f, TranscriptionFrame)]

    def interim_texts(self) -> list[str]:
        return [f.text for f in self.pushed if isinstance(f, InterimTranscriptionFrame)]


@pytest.mark.asyncio
async def test_multi_segment_finalization_becomes_one_frame():
    """A finalization with several segments must push exactly one finalized frame.

    One finalized frame per segment makes turn-taking close the user turn on the
    first frame, so the remaining segments of the utterance are dropped.
    """
    service = _RecordingService()
    await service._send_frames(
        [_segment("One."), _segment("Two."), _segment("Three.")],
        finalized=True,
    )

    assert service.finalized_texts() == ["One. Two. Three."]


@pytest.mark.asyncio
async def test_single_segment_finalization_is_unchanged():
    service = _RecordingService()
    await service._send_frames([_segment("Hello there.")], finalized=True)

    assert service.finalized_texts() == ["Hello there."]


@pytest.mark.asyncio
async def test_interim_segments_are_not_coalesced():
    service = _RecordingService()
    await service._send_frames([_segment("One."), _segment("Two.")], finalized=False)

    assert service.interim_texts() == ["One.", "Two."]
    assert service.finalized_texts() == []


@pytest.mark.asyncio
async def test_different_speakers_stay_separate():
    service = _RecordingService()
    await service._send_frames(
        [
            _segment("Hello.", speaker_id="S1"),
            _segment("Hi, how can I help?", speaker_id="S2"),
            _segment("I have a question.", speaker_id="S1"),
        ],
        finalized=True,
    )

    assert service.finalized_texts() == [
        "Hello.",
        "Hi, how can I help?",
        "I have a question.",
    ]


def test_coalesce_merges_results_and_eou():
    service = _RecordingService()
    merged = service._coalesce_segments(
        [
            _segment("One.", is_eou=False),
            _segment("Two.", is_eou=True),
        ]
    )

    assert len(merged) == 1
    assert merged[0]["text"] == "One. Two."
    assert merged[0]["is_eou"] is True
    assert len(merged[0]["results"]) == 2
    assert merged[0]["timestamp"] == "2026-01-01T00:00:00.000Z"
