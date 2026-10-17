from __future__ import annotations

import importlib
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


ATTENTION_OUTPUT = "mrte_cross_attn_softmax"

# Symbols the Japanese frontend adds after a phoneme; they are not phonemes.
_PROSODY_MARKS = frozenset("#[]")
# Every Japanese mora ends in a vowel, the moraic nasal or the geminate stop.
_MORA_ENDS = frozenset(("a", "i", "u", "e", "o", "N", "cl"))
_SMALL_KANA = frozenset("ぁぃぅぇぉゃゅょゎァィゥェォャュョヮ")
# What the English frontend removes from its own output before returning it.
_ENGLISH_DROPPED = frozenset((" ", "<pad>", "UW", "</s>", "<s>"))
# Phones the voice model reads as a pause.  "_" is the pause inside a Japanese phrase.
_PAUSES = frozenset(("!", "?", "…", ",", ".", "-", "SP", "SP2", "SP3", "_"))
# What the phone path holds for a frame that belongs to no phone.
SILENCE = -1


def monotonic_phone_path(
    attention: Any,
    force_complete: bool = True,
    pauses: Optional[Sequence[bool]] = None,
) -> np.ndarray:
    """Return a left-to-right phone assignment for every acoustic frame.

    GPT-SoVITS exposes one attention matrix per MRTE head.  Independent
    argmaxes can jump backwards, which in turn creates overlapping or reversed
    character timestamps.  This dynamic program allows a frame to stay on the
    current phone or advance to the next one.

    A silent frame does not attend to the phone that is being spoken.  Most of
    its attention goes to the last phone of the text and some to the pause
    phones, wherever those stand.  Read column by column, the silence before
    the first word looks like the end of the text, and the path runs through
    every phone to get there.  Silence therefore has a score of its own, taken
    from the attention on the last phone and on the phones marked in
    ``pauses``.  Pause phones are scored with it, and a silence may come before
    any other phone; its frames are returned as ``SILENCE``.

    ``force_complete`` is appropriate for a completed utterance.  Incremental
    streaming snapshots leave it disabled so an unfinished prefix is not
    stretched across phones that have not been spoken yet.
    """

    scores = _attention_scores(attention)
    if scores.size == 0:
        return np.zeros((0,), dtype=np.int64)

    frame_count, phone_count = scores.shape
    if phone_count == 1:
        return np.zeros((frame_count,), dtype=np.int64)

    pause = np.zeros((phone_count,), dtype=bool)
    if pauses is not None:
        marked = np.asarray(list(pauses), dtype=bool)[:phone_count]
        pause[: len(marked)] = marked
    silent = pause.copy()
    silent[-1] = True
    # The last phone keeps a quarter to half of the attention while a voice is
    # heard and takes two thirds or more in silence.  Squared, the share stays
    # below the phone being spoken until it is clearly the larger part.
    silence = scores[:, silent].sum(axis=1) ** 2

    # Two states per phone: the silence that may come before it, then the
    # phone.  Next to a pause phone the pause itself is that silence.
    gap = ~pause
    gap[1:] &= ~pause[:-1]
    state_count = 2 * phone_count
    log_scores = np.full((frame_count, state_count), -np.inf, dtype=np.float64)
    log_scores[:, 1::2] = np.log(np.maximum(np.where(pause[None, :], silence[:, None], scores), 1e-12))
    log_scores[:, np.flatnonzero(gap) * 2] = np.log(np.maximum(silence, 1e-12))[:, None]
    is_phone = np.arange(state_count) % 2 == 1

    dp = np.full((state_count,), -np.inf, dtype=np.float64)
    dp[:2] = log_scores[0, :2]
    steps = np.zeros((frame_count, state_count), dtype=np.int8)
    for frame_index in range(1, frame_count):
        one = np.concatenate(([-np.inf], dp[:-1]))
        # A phone is also reached from the phone before it, without a silence.
        two = np.where(is_phone, np.concatenate(([-np.inf, -np.inf], dp[:-2])), -np.inf)
        take_one = one > dp
        best = np.where(take_one, one, dp)
        take_two = two > best
        steps[frame_index] = np.where(take_two, 2, take_one)
        dp = np.where(take_two, two, best) + log_scores[frame_index]

    if force_complete and frame_count >= phone_count and np.isfinite(dp[-1]):
        state = state_count - 1
    else:
        state = int(np.argmax(dp))

    result = np.zeros((frame_count,), dtype=np.int64)
    for frame_index in range(frame_count - 1, -1, -1):
        result[frame_index] = state // 2 if state % 2 else SILENCE
        state -= int(steps[frame_index, state])
    return result


def process_attention(
    attention: Any,
    word2ph_batch: Sequence[Sequence[int]],
    semantic_lens: Sequence[int],
    speed_factor: float,
    phone_symbols_batch: Sequence[Sequence[str]],
    force_complete: bool = True,
) -> List[List[Dict[str, Any]]]:
    """Convert MRTE cross-attention into monotonic character timestamps.

    A character lasts from the first to the last frame of its phones.  The
    silence before the first character belongs to no character, so the first
    one does not start at zero.  A pause belongs to its punctuation mark;
    where the text has none, it is a gap between two characters.
    """

    attn = _attention_array(attention)
    if attn.ndim != 4 or attn.shape[0] < 1:
        return []

    attn_mean = attn.mean(axis=1)[0]  # [TotalSemanticLen, TotalPhoneLen]
    frame_duration = 0.02 / (speed_factor if speed_factor > 0 else 1.0)

    timestamp_results: List[List[Dict[str, Any]]] = []
    current_semantic_offset = 0
    current_phone_offset = 0

    for item_index, word2ph in enumerate(word2ph_batch):
        sem_len = int(semantic_lens[item_index]) if item_index < len(semantic_lens) else 0
        phone_len = int(sum(max(0, int(x)) for x in word2ph))
        phone_symbols = phone_symbols_batch[item_index] if item_index < len(phone_symbols_batch) else []

        item_attn = attn_mean[current_semantic_offset : current_semantic_offset + sem_len, :]
        item_attn = item_attn[:, current_phone_offset : current_phone_offset + phone_len]
        pauses = [symbol in _PAUSES for symbol in phone_symbols[:phone_len]]
        phone_indices = monotonic_phone_path(item_attn, force_complete=force_complete, pauses=pauses)

        phone_to_frames: Dict[int, List[int]] = {}
        for frame_idx, ph_idx in enumerate(phone_indices):
            if ph_idx != SILENCE:
                phone_to_frames.setdefault(int(ph_idx), []).append(frame_idx)
        last_observed_phone = int(phone_indices.max()) if phone_indices.size else -1

        char_timestamps: List[Dict[str, Any]] = []
        current_ph_idx = 0

        for char_idx, num_phones in enumerate(word2ph):
            num_phones = max(0, int(num_phones))
            start_ph = current_ph_idx
            end_ph = min(phone_len, current_ph_idx + num_phones)
            if not force_complete and start_ph > last_observed_phone:
                break
            current_phones = [
                phone_symbols[p_idx] if p_idx < len(phone_symbols) else ""
                for p_idx in range(start_ph, end_ph)
            ]

            frames: List[int] = []
            for phone_idx in range(start_ph, end_ph):
                frames.extend(phone_to_frames.get(phone_idx, []))

            if frames:
                start_frame = min(frames)
                end_frame = max(frames) + 1
                start_time = start_frame * frame_duration
                end_time = end_frame * frame_duration
            else:
                start_time = char_timestamps[-1]["end"] if char_timestamps else 0.0
                end_time = start_time

            char_timestamps.append(
                {
                    "char_idx": char_idx,
                    "ph_start": start_ph,
                    "ph_end": end_ph,
                    "phones": current_phones,
                    "start": float(start_time),
                    "end": float(end_time),
                }
            )
            current_ph_idx += num_phones

        timestamp_results.append(char_timestamps)
        current_semantic_offset += sem_len
        current_phone_offset += phone_len

    return timestamp_results


def finalize_timestamps(
    timestamps: Iterable[Dict[str, Any]],
    norm_text: str,
    offset_seconds: float = 0.0,
    original_text: Optional[str] = None,
    char_offset: int = 0,
) -> List[Dict[str, Any]]:
    """Map normalized character timing back to stable source-text indices.

    The old implementation exposed normalized indices directly.  That works
    for simple Japanese strings, but breaks when normalization removes a
    character and when multiple synthesis segments reset indices to zero.
    Keeping the public record shape while mapping onto the original input lets
    both full and streaming synthesis share the same timeline contract.
    """

    source_text = norm_text if original_text is None else original_text
    index_map = _normalized_to_original_indices(norm_text, source_text)
    merged: Dict[int, Dict[str, Any]] = {}
    reached_end = False

    for item in timestamps:
        norm_idx = int(item.get("char_idx", 0))
        if norm_idx < 0 or norm_idx >= len(index_map):
            continue
        reached_end = reached_end or norm_idx == len(index_map) - 1
        source_idx = index_map[norm_idx]
        if source_idx < 0 or source_idx >= len(source_text):
            continue

        absolute_idx = char_offset + source_idx
        value = {
            "char_idx": absolute_idx,
            "ph_start": int(item.get("ph_start", 0)),
            "ph_end": int(item.get("ph_end", 0)),
            "phones": list(item.get("phones", [])),
            "start": float(item.get("start", 0.0)) + offset_seconds,
            "end": float(item.get("end", 0.0)) + offset_seconds,
            "char": source_text[source_idx],
        }

        previous = merged.get(absolute_idx)
        if previous is None:
            merged[absolute_idx] = value
        else:
            previous["ph_start"] = min(previous["ph_start"], value["ph_start"])
            previous["ph_end"] = max(previous["ph_end"], value["ph_end"])
            previous["phones"].extend(value["phones"])
            previous["start"] = min(previous["start"], value["start"])
            previous["end"] = max(previous["end"], value["end"])

    if merged:
        # A character the normalization dropped ("……" is read as one "…") is
        # not spoken, but it has a place in the text: where the character
        # before it ends.  Past the last timed character only once the text
        # has been timed to its end.
        stop = char_offset + len(source_text) if reached_end else max(merged)
        before = None
        for absolute_idx in range(char_offset, stop):
            if absolute_idx in merged:
                before = merged[absolute_idx]
                continue
            moment = before["end"] if before is not None else merged[min(merged)]["start"]
            phone = before["ph_end"] if before is not None else merged[min(merged)]["ph_start"]
            before = merged[absolute_idx] = {
                "char_idx": absolute_idx,
                "ph_start": phone,
                "ph_end": phone,
                "phones": [],
                "start": moment,
                "end": moment,
                "char": source_text[absolute_idx - char_offset],
            }

    return [merged[index] for index in sorted(merged)]


def scale_timestamps_to_duration(
    timestamps: Iterable[Dict[str, Any]],
    duration_seconds: float,
    offset_seconds: float = 0.0,
) -> List[Dict[str, Any]]:
    """Scale a relative alignment so its end matches emitted PCM exactly."""

    result = [_copy_timestamp(item) for item in timestamps]
    if not result:
        return result
    timeline_end = max(float(item.get("end", offset_seconds)) for item in result)
    relative_end = timeline_end - float(offset_seconds)
    if relative_end <= 1e-9:
        return result
    scale = max(0.0, float(duration_seconds)) / relative_end
    for item in result:
        item["start"] = float(offset_seconds) + (
            float(item.get("start", offset_seconds)) - float(offset_seconds)
        ) * scale
        item["end"] = float(offset_seconds) + (
            float(item.get("end", offset_seconds)) - float(offset_seconds)
        ) * scale
    return result


class StreamingTimeline:
    """Character timing for one text segment whose audio is still being generated.

    The voice model is evaluated again every time more of the segment exists.
    An evaluation that has seen little more than the silence before the first
    word cannot place the characters, and a later one places them better, so
    a character is not fixed when it first appears in an alignment.  It is
    handed out once, from the newest alignment, when the audio that holds its
    start is emitted.
    """

    def __init__(
        self,
        word2ph: Sequence[int],
        phone_symbols: Sequence[str],
        norm_text: str,
        speed_factor: float = 1.0,
        offset_seconds: float = 0.0,
        original_text: Optional[str] = None,
        char_offset: int = 0,
    ) -> None:
        self._word2ph = list(word2ph)
        self._phone_symbols = list(phone_symbols)
        self._norm_text = norm_text
        self._speed = speed_factor if speed_factor > 0 else 1.0
        self._offset = float(offset_seconds)
        self._original_text = original_text
        self._char_offset = int(char_offset)
        self._last_char = -1
        self._last_end = self._offset

    def collect(self, attention: Any, frame_count: int, final: bool = False) -> List[Dict[str, Any]]:
        """Return the characters that start in the first ``frame_count`` emitted audio frames.

        ``attention`` is the newest evaluation of the segment and ``frame_count``
        counts 20 ms audio frames.  Every character is returned once, in text
        order, and none starts before the previous one ended.  ``final``
        returns all that remain.
        """

        scores = _attention_scores(attention)
        if not scores.size:
            return []
        rows = process_attention(
            scores,
            [self._word2ph],
            [len(scores)],
            self._speed,
            [self._phone_symbols],
            force_complete=final,
        )[0]
        emitted = self._offset + frame_count * 0.02
        known = self._offset + len(scores) * 0.02 / self._speed

        collected: List[Dict[str, Any]] = []
        for row in finalize_timestamps(
            rows, self._norm_text, self._offset, original_text=self._original_text, char_offset=self._char_offset
        ):
            if row["char_idx"] <= self._last_char:
                continue
            if final:
                row["end"] = min(row["end"], emitted)
            elif row["start"] >= emitted - 1e-6 or row["end"] >= known - 1e-6:
                # Not audible yet, or still being spoken at the edge of what exists.
                break
            row["start"] = max(row["start"], self._last_end)
            row["end"] = max(row["end"], row["start"])
            self._last_char = row["char_idx"]
            self._last_end = row["end"]
            collected.append(row)
        return collected


def japanese_word2ph(norm_text: str, phones: Sequence[str]) -> List[int]:
    """Return how many of ``phones`` each character of ``norm_text`` produced.

    The Japanese frontend returns one flat phone list.  Taking every phone for
    a character maps the first half of the phones onto the whole sentence, so
    the characters finish long before the voice does.

    Word boundaries come from Open JTalk and are exact.  Inside a word, kana
    are placed by mora; kanji and other characters share the rest of the word
    evenly, which can be off by a mora.
    """

    if not norm_text:
        return []
    japanese = importlib.import_module("text.japanese")

    # The frontend spells out some symbols and lower-cases before splitting.
    pieces = [japanese.symbols_to_japanese(char).lower() for char in norm_text]
    text = "".join(pieces)
    origin = [index for index, piece in enumerate(pieces) for _ in piece]

    counts = [0] * len(norm_text)
    total = 0
    position = 0
    for mark in list(japanese._japanese_marks.finditer(text)) + [None]:
        end = len(text) if mark is None else mark.start()
        if end > position:
            chunk = text[position:end]
            symbols = japanese.pyopenjtalk_g2p_prosody(chunk)[1:-1]
            chunk_counts = _japanese_chunk_counts(japanese.pyopenjtalk, chunk, symbols)
            for offset, count in enumerate(chunk_counts):
                counts[origin[position + offset]] += count
            total += len(symbols)
        if mark is not None:
            if mark.group() != " ":
                counts[origin[mark.start()]] += 1
                total += 1
            position = mark.end()

    if total != len(phones) or text != japanese.symbols_to_japanese(norm_text).lower():
        # The frontend no longer builds its phones the way this function reads
        # them.  The timeline must not stop speech, so it turns approximate.
        print(
            f"Japanese timeline: expected {total} phones for {norm_text!r}, "
            f"the frontend produced {len(phones)}; characters are spread evenly."
        )
        return _spread(len(phones), len(norm_text))
    return counts


def english_word2ph(norm_text: str, phones: Sequence[str]) -> List[int]:
    """Return how many of ``phones`` each character of ``norm_text`` produced.

    The English frontend pronounces word by word, so the phones of a word are
    known exactly.  Which letter stands for which sound is not: a word's
    phones are spread evenly along its letters, starting with the first.
    """

    if not norm_text:
        return []
    english = importlib.import_module("text.english")

    # The frontend puts a space between the words it pronounced.
    words: List[List[str]] = []
    pronounced = english._g2p(norm_text)
    if pronounced:
        words.append([])
    for phone in pronounced:
        if phone == " ":
            words.append([])
        else:
            words[-1].append(phone)
    tokens = english.word_tokenize(norm_text)

    counts = [0] * len(norm_text)
    total = 0
    position = 0
    aligned = len(tokens) == len(words)
    for token, word in zip(tokens, words) if aligned else ():
        start = norm_text.find(token, position)
        if start < 0:
            aligned = False
            break
        kept = 0
        for phone in word:
            if phone in _ENGLISH_DROPPED:
                continue
            if phone == "<unk>" or phone == "'" or phone in english.symbols:
                kept += 1
        letters = [index for index, char in enumerate(token) if char.isalpha()] or list(range(len(token)))
        for index, count in zip(letters, _spread(kept, len(letters))):
            counts[start + index] += count
        total += kept
        position = start + len(token)
    # The cleaner puts a comma in front of a very short phone list.
    if total < 4:
        counts[0] += 1
        total += 1

    if not aligned or total != len(phones):
        # Same reasoning as for Japanese: approximate rather than fail.
        print(
            f"English timeline: expected {total} phones for {norm_text!r}, "
            f"the frontend produced {len(phones)}; characters are spread evenly."
        )
        return _spread(len(phones), len(norm_text))
    return counts


def _japanese_chunk_counts(jtalk: Any, chunk: str, symbols: Sequence[str]) -> List[int]:
    """Phone counts for a run of text without punctuation."""

    # One entry per phoneme: its symbol and how many symbols it spans together
    # with the prosody marks that follow it.
    phonemes: List[List[Any]] = []
    for symbol in symbols:
        if symbol in _PROSODY_MARKS and phonemes:
            phonemes[-1][1] += 1
        else:
            phonemes.append([symbol, 1])

    words = jtalk.run_frontend(chunk)
    # Phonemes are produced word by word, so a shorter word list yields a
    # prefix of them.  Two of the labels are the leading and trailing silence.
    ends = [max(0, len(jtalk.make_label(words[: index + 1])) - 2) for index in range(len(words))]
    # A pause only shows up once the next word arrives; it belongs to the
    # symbol that asked for it ("・").
    for index in range(len(ends) - 1):
        while ends[index] < min(ends[index + 1], len(phonemes)) and phonemes[ends[index]][0] == "_":
            ends[index] += 1
    spelling = "".join(word["string"] for word in words)
    spelled: List[int] = []
    start = 0
    for word, end in zip(words, ends):
        spelled += _japanese_word_counts(word["string"], phonemes[start:end])
        start = max(start, end)
    if not spelling or start != len(phonemes) or sum(spelled) != len(symbols):
        return _spread(len(symbols), len(chunk))

    # Open JTalk respells some text, digits above all ("1000" becomes "千").
    # What it kept maps one to one; a rewritten stretch shares its phones.
    counts = [0] * len(chunk)
    matcher = SequenceMatcher(None, _fold_width(chunk), _fold_width(spelling), autojunk=False)
    for tag, chunk_start, chunk_end, spelled_start, spelled_end in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(chunk_end - chunk_start):
                counts[chunk_start + offset] += spelled[spelled_start + offset]
        elif chunk_end > chunk_start:
            shared = _spread(sum(spelled[spelled_start:spelled_end]), chunk_end - chunk_start)
            for offset, count in enumerate(shared):
                counts[chunk_start + offset] += count
        else:
            counts[max(0, chunk_start - 1)] += sum(spelled[spelled_start:spelled_end])
    return counts


def _japanese_word_counts(string: str, phonemes: Sequence[Sequence[Any]]) -> List[int]:
    """Phone counts for the characters of one word."""

    if not string:
        return []
    moras: List[List[int]] = []
    current: List[int] = []
    for symbol, size in phonemes:
        current.append(size)
        if symbol in _MORA_ENDS:
            moras.append(current)
            current = []
    if current:
        if moras:
            moras[-1].extend(current)
        else:
            moras.append(current)

    # A small kana normally shares the mora of the kana before it ("きゃ"),
    # but it can also stand for a mora of its own ("あぁ").
    claims = _mora_claims(string, len(moras), True) or _mora_claims(string, len(moras), False)
    if claims is None:
        return _spread(sum(size for _, size in phonemes), len(string))

    counts = [0] * len(string)
    taken = 0
    index = 0
    while index < len(string):
        sizes = [size for mora in moras[taken : taken + claims[index]] for size in mora]
        taken += claims[index]
        riders = index + 1
        while riders < len(string) and claims[riders] == 0:
            riders += 1
        # Characters sharing a mora take one phoneme each, the last the rest.
        for offset in range(riders - index - 1):
            counts[index + offset] = sizes[offset] if offset < len(sizes) else 0
        counts[riders - 1] = sum(sizes[riders - index - 1 :])
        index = riders
    return counts


def _mora_claims(string: str, mora_count: int, merge_small_kana: bool) -> Optional[List[int]]:
    """Moras per character: one per kana, the rest shared by the other characters."""

    claims: List[int] = []
    others: List[int] = []
    for index, char in enumerate(string):
        if _is_kana(char):
            rides = merge_small_kana and char in _SMALL_KANA and index > 0 and _is_kana(string[index - 1])
            claims.append(0 if rides else 1)
        else:
            claims.append(0)
            others.append(index)
    rest = mora_count - sum(claims)
    if rest < len(others) or (rest > 0 and not others):
        return None
    for position, index in enumerate(others):
        claims[index] = rest // len(others) + (1 if position < rest % len(others) else 0)
    return claims


def _is_kana(char: str) -> bool:
    return "ぁ" <= char <= "ゖ" or "ァ" <= char <= "ヺ" or char == "ー"


def _fold_width(text: str) -> str:
    """Make full-width and half-width forms compare equal, one character each."""

    folded = []
    for char in text:
        normalized = unicodedata.normalize("NFKC", char)
        folded.append(normalized if len(normalized) == 1 else char)
    return "".join(folded)


def _spread(total: int, count: int) -> List[int]:
    """Share ``total`` among ``count`` slots evenly along their order, the first slot first."""

    marks = [-(-index * total // count) for index in range(count + 1)]
    return [marks[index + 1] - marks[index] for index in range(count)]


def _attention_array(value: Any) -> np.ndarray:
    result = _to_numpy(value)
    if result.ndim == 2:
        result = result[None, None, :, :]
    elif result.ndim == 3:
        result = result[None, :, :, :]
    return result


def _attention_scores(value: Any) -> np.ndarray:
    result = _to_numpy(value)
    if result.ndim == 4:
        result = result.mean(axis=(0, 1))
    elif result.ndim == 3:
        result = result.mean(axis=0)
    if result.ndim != 2:
        return np.zeros((0, 0), dtype=np.float64)
    return np.asarray(result, dtype=np.float64)


def _normalized_to_original_indices(norm_text: str, original_text: str) -> List[int]:
    if not norm_text:
        return []
    if norm_text == original_text:
        return list(range(len(norm_text)))

    mapping = [-1] * len(norm_text)
    matcher = SequenceMatcher(None, norm_text, original_text, autojunk=False)
    for tag, norm_start, norm_end, source_start, source_end in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(norm_end - norm_start):
                mapping[norm_start + offset] = source_start + offset
        elif tag in ("replace", "delete"):
            norm_length = norm_end - norm_start
            source_length = source_end - source_start
            for offset in range(norm_length):
                if source_length > 0:
                    mapped_offset = min(source_length - 1, int(offset * source_length / max(1, norm_length)))
                    mapping[norm_start + offset] = source_start + mapped_offset
                elif original_text:
                    mapping[norm_start + offset] = min(len(original_text) - 1, source_start)

    last_valid = 0
    for index, value in enumerate(mapping):
        if value < 0:
            mapping[index] = last_valid
        else:
            last_valid = value
    for index in range(len(mapping) - 2, -1, -1):
        if mapping[index] < 0:
            mapping[index] = mapping[index + 1]
    return mapping


def _copy_timestamp(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "char_idx": int(item.get("char_idx", 0)),
        "ph_start": int(item.get("ph_start", 0)),
        "ph_end": int(item.get("ph_end", 0)),
        "phones": list(item.get("phones", [])),
        "start": float(item.get("start", 0.0)),
        "end": float(item.get("end", 0.0)),
        "char": str(item.get("char", "")),
    }


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)
