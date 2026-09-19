import math

import numpy as np

from oat_notes.speaker_store import SpeakerStore
from oat_notes.types import Channel
from oat_notes.unknown_turns import MEMORY_LIMIT, UnknownTurn, backfill_score, remember


def unit(x, y):
    length = math.hypot(x, y)
    return np.array([x / length, y / length], dtype=np.float32)


def test_memory_keeps_the_newest_turns_and_copies_the_embedding():
    turns = {}
    source = np.array([1.0, 0.0], dtype=np.float32)
    for line_id in range(MEMORY_LIMIT + 5):
        remember(turns, line_id, Channel.MIC, source, 2.0)
    source[0] = 0.0
    assert len(turns) == MEMORY_LIMIT
    assert 4 not in turns and MEMORY_LIMIT + 4 in turns
    assert turns[MEMORY_LIMIT + 4].embedding[0] == 1.0


def test_backfill_needs_a_clear_own_match_and_no_close_rival(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    sarah = store.create_speaker("Sarah")
    dev = store.create_speaker("Dev")
    store.add_sample(sarah.speaker_id, unit(1.0, 0.0), Channel.MIC, 5.0, 1.0)
    store.add_sample(dev.speaker_id, unit(0.0, 1.0), Channel.MIC, 5.0, 1.0)

    clear = UnknownTurn(Channel.MIC, unit(1.0, 0.1), 2.0)
    assert backfill_score(clear, store, sarah.speaker_id, [dev.speaker_id]) > 0.9

    weak = UnknownTurn(Channel.MIC, unit(0.5, 1.0), 2.0)
    assert backfill_score(weak, store, sarah.speaker_id, [dev.speaker_id]) is None

    contested = UnknownTurn(Channel.MIC, unit(1.0, 0.95), 2.0)
    assert backfill_score(contested, store, sarah.speaker_id, [dev.speaker_id]) is None
    assert backfill_score(contested, store, sarah.speaker_id, []) > 0.7


def test_backfill_without_a_profile_is_none(tmp_path):
    store = SpeakerStore(tmp_path / "speakers.db")
    nobody = store.create_speaker("Nobody")
    turn = UnknownTurn(Channel.MIC, unit(1.0, 0.0), 2.0)
    assert backfill_score(turn, store, nobody.speaker_id, []) is None
