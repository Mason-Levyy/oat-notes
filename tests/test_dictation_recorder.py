import numpy as np
import pytest

from oat_notes.config import Config
from oat_notes.dictation.recorder import UtteranceRecorder
from oat_notes.transcriber import NO_HINTS, Transcriber
from oat_notes.types import Channel, TranscriptSegment

WINDOW = 512
CONFIG = Config(
    vad_window_samples=WINDOW,
    dictation_silence_split_seconds=0.05,
    dictation_clause_gap_seconds=0.25,
    dictation_sentence_gap_seconds=0.3,
    min_speech_seconds=0.02,
    pre_roll_windows=1,
    dictation_max_chunk_seconds=0.5,
)


class EnergyFakeVad:
    """Speech wherever the samples are loud — lets tests shape audio directly."""

    window_samples = WINDOW

    def __init__(self):
        self.resets = 0

    def speech_probability(self, window):
        return 1.0 if float(np.abs(window).max()) > 0.5 else 0.0

    def reset(self):
        self.resets += 1


class CountingTranscriber(Transcriber):
    def __init__(self, texts=None):
        self.calls = 0
        self.hints = []
        self._texts = list(texts) if texts else None

    def transcribe(self, chunk, hints=NO_HINTS):
        self.hints.append(hints)
        texts = self._texts or [f"chunk{self.calls}"]
        text = texts[self.calls % len(texts)]
        self.calls += 1
        return TranscriptSegment(
            text=text, channel=chunk.channel, start=chunk.start, end=chunk.end
        )


class ExplodingTranscriber(Transcriber):
    def transcribe(self, chunk, hints=NO_HINTS):
        raise RuntimeError("model fell over")


def speech(windows, amplitude=0.75):
    return np.full(windows * WINDOW, amplitude, dtype=np.float32)


def silence(windows):
    return np.zeros(windows * WINDOW, dtype=np.float32)


def make_recorder(transcriber=None, level_sink=None, hotwords=None):
    return UtteranceRecorder(
        CONFIG,
        transcriber or CountingTranscriber(),
        level_sink=level_sink,
        vad=EnergyFakeVad(),
        hotwords=hotwords,
    )


def feed(recorder, blocks):
    timestamp = 0.0
    for block in blocks:
        recorder.frame_queue.put((Channel.MIC, timestamp, block))
        timestamp += block.size / CONFIG.sample_rate


def test_utterance_joins_chunks_in_order():
    recorder = make_recorder(CountingTranscriber(["hello there", "second part"]))
    recorder.start_worker()
    feed(recorder, [speech(6), silence(6), speech(6), silence(6)])
    assert recorder.stop() == "hello there second part"


def test_a_breath_between_chunks_does_not_end_the_sentence():
    recorder = make_recorder(CountingTranscriber(["So I think.", "We should ship."]))
    recorder.start_worker()
    feed(recorder, [speech(6), silence(6), speech(6), silence(6)])
    assert recorder.stop() == "So I think we should ship."


def test_a_long_silence_between_chunks_ends_the_sentence():
    recorder = make_recorder(CountingTranscriber(["Let us ship it", "QA is done."]))
    recorder.start_worker()
    quiet = int(CONFIG.dictation_sentence_gap_seconds / CONFIG.window_seconds) + 2
    feed(recorder, [speech(6), silence(quiet), speech(6), silence(quiet)])
    assert recorder.stop() == "Let us ship it. QA is done."


def test_each_chunk_hears_what_came_before_it_and_the_vocabulary():
    transcriber = CountingTranscriber(["hello there", "second part"])
    recorder = make_recorder(transcriber, hotwords="Mason, Oat Notes")
    recorder.start_worker()
    feed(recorder, [speech(6), silence(6), speech(6), silence(6)])
    recorder.stop()
    first, second = transcriber.hints
    assert (first.prompt, first.hotwords) == (None, "Mason, Oat Notes")
    assert (second.prompt, second.hotwords) == ("hello there", "Mason, Oat Notes")


def test_the_prompt_is_only_the_tail_of_a_long_utterance():
    transcriber = CountingTranscriber(["x" * 300, "tail"])
    recorder = make_recorder(transcriber)
    recorder.start_worker()
    feed(recorder, [speech(6), silence(6), speech(6), silence(6)])
    recorder.stop()
    assert len(transcriber.hints[1].prompt) == CONFIG.dictation_prompt_chars


def test_the_prompt_resets_between_utterances():
    transcriber = CountingTranscriber(["first", "second"])
    recorder = make_recorder(transcriber)
    recorder.start_worker()
    feed(recorder, [speech(6), silence(6)])
    recorder.stop()
    recorder.start_worker()
    feed(recorder, [speech(6), silence(6)])
    recorder.stop()
    assert [hints.prompt for hints in transcriber.hints] == [None, None]


def test_tail_chunk_is_flushed_on_stop():
    transcriber = CountingTranscriber()
    recorder = make_recorder(transcriber)
    recorder.start_worker()
    feed(recorder, [speech(6)])
    assert recorder.stop() == "chunk0"
    assert transcriber.calls == 1


def test_silence_only_produces_no_text():
    recorder = make_recorder()
    recorder.start_worker()
    feed(recorder, [silence(20)])
    assert recorder.stop() == ""


def test_cancel_discards_text_and_skips_the_tail():
    transcriber = CountingTranscriber()
    recorder = make_recorder(transcriber)
    recorder.start_worker()
    feed(recorder, [speech(6), silence(6), speech(6)])
    recorder.cancel()
    assert recorder.stop() == ""
    assert transcriber.calls <= 1


def test_transcription_error_does_not_lose_the_rest():
    recorder = make_recorder(ExplodingTranscriber())
    recorder.start_worker()
    feed(recorder, [speech(6), silence(6)])
    assert recorder.stop() == ""


def test_speech_seconds_tracks_voiced_audio():
    recorder = make_recorder()
    recorder.start_worker()
    feed(recorder, [speech(10), silence(4)])
    recorder.stop()
    assert recorder.speech_seconds == pytest.approx(10 * CONFIG.window_seconds)


def test_quiet_tap_reports_no_speech():
    recorder = make_recorder()
    recorder.start_worker()
    feed(recorder, [silence(8)])
    recorder.stop()
    assert recorder.speech_seconds == 0.0


def test_level_sink_receives_rms_per_block():
    levels = []
    recorder = make_recorder(level_sink=levels.append)
    recorder.start_worker()
    feed(recorder, [speech(2, amplitude=0.5), silence(2)])
    recorder.stop()
    assert levels == [0.5, 0.0]


def test_broken_level_sink_does_not_stall_capture():
    def explode(_level):
        raise RuntimeError("meter broke")

    recorder = make_recorder(level_sink=explode)
    recorder.start_worker()
    feed(recorder, [speech(6), silence(6)])
    assert recorder.stop() == "chunk0"


def test_restart_resets_state_between_utterances():
    recorder = make_recorder(CountingTranscriber(["first", "second"]))
    recorder.start_worker()
    feed(recorder, [speech(6), silence(6)])
    assert recorder.stop() == "first"

    recorder.start_worker()
    feed(recorder, [speech(6), silence(6)])
    assert recorder.stop() == "second"
    assert recorder.speech_seconds == pytest.approx(6 * CONFIG.window_seconds)


def test_restart_resets_the_vad_session():
    recorder = make_recorder()
    recorder.start_worker()
    recorder.stop()
    recorder.start_worker()
    recorder.stop()
    assert recorder._vad.resets == 2


def test_long_speech_force_splits_so_work_starts_early():
    transcriber = CountingTranscriber()
    recorder = make_recorder(transcriber)
    recorder.start_worker()
    windows_per_chunk = int(CONFIG.dictation_max_chunk_seconds / CONFIG.window_seconds)
    feed(recorder, [speech(windows_per_chunk * 3)])
    recorder.stop()
    assert transcriber.calls >= 3
