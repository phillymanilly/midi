#!/usr/bin/env python3
import argparse
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pretty_midi


# ----------------------------
# Music helpers
# ----------------------------

NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name_to_pc(name: str) -> int:
    name = name.strip().upper()
    flats = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#"}
    name = flats.get(name, name)
    if name not in NOTE_NAMES_SHARP:
        raise ValueError(f"Unknown note name: {name}")
    return NOTE_NAMES_SHARP.index(name)


def midi_note(pc: int, octave: int) -> int:
    return 12 * (octave + 1) + (pc % 12)


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def normalize_artist(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower().strip())


# ----------------------------
# Chords / voicings
# ----------------------------

CHORD_QUALITIES: Dict[str, List[int]] = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "m7b5": [0, 3, 6, 10],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "9": [0, 4, 7, 10, 14],
    "maj9": [0, 4, 7, 11, 14],
    "min9": [0, 3, 7, 10, 14],
    "11": [0, 4, 7, 10, 14, 17],
    "min11": [0, 3, 7, 10, 14, 17],
    "13": [0, 4, 7, 10, 14, 17, 21],
    "7alt": [0, 4, 10, 13, 15],
}


def build_chord_pcs(root_pc: int, quality: str) -> List[int]:
    intervals = CHORD_QUALITIES[quality]
    return [(root_pc + interval) % 12 for interval in intervals]


def voice_chord(
    pcs: List[int],
    root_pc: int,
    base_octave: int,
    spread: str,
    inversion: int,
    drop2: bool,
    keep_root: bool = True,
) -> List[int]:
    if keep_root and root_pc not in pcs:
        pcs = [root_pc] + pcs

    ordered = sorted(pcs, key=lambda pc: (pc - root_pc) % 12)
    if ordered:
        inv = inversion % len(ordered)
        ordered = ordered[inv:] + ordered[:inv]

    notes = []
    current_octave = base_octave
    last_midi = None
    for pc in ordered:
        note = midi_note(pc, current_octave)
        if last_midi is not None:
            while note <= last_midi:
                current_octave += 1
                note = midi_note(pc, current_octave)
        notes.append(note)
        last_midi = note

    if spread == "spread":
        if len(notes) >= 4:
            notes[-1] += 12
        if len(notes) >= 5:
            notes[-2] += 12

    if drop2 and len(notes) >= 4:
        notes[-2] -= 12
        notes = sorted(notes)

    return notes


# ----------------------------
# Style profiles
# ----------------------------


@dataclass
class StyleProfile:
    name: str
    bpm_range: Tuple[int, int]
    key_pool: List[str]
    progression_pool: List[List[Tuple[int, str]]]
    rhythm_pool: List[List[Tuple[float, float, float]]]
    chord_octave: Tuple[int, int]
    bass_octave: Tuple[int, int]
    spread_weights: Dict[str, float]
    drop2_prob: float
    inversion_max: int
    ext_velocity: Tuple[int, int]
    core_velocity: Tuple[int, int]
    humanize_ms: int
    swing: float


def degrees_to_pc(key_root_pc: int, is_minor: bool, degree: int) -> int:
    major_steps = [0, 2, 4, 5, 7, 9, 11]
    minor_steps = [0, 2, 3, 5, 7, 8, 10]
    steps = minor_steps if is_minor else major_steps
    return (key_root_pc + steps[(degree - 1) % 7]) % 12


def parse_key(key: str) -> Tuple[int, bool]:
    match = re.match(r"^([A-G])([#B]?)(MIN|MAJ)$", key.strip().upper())
    if not match:
        raise ValueError(f"Bad key format: {key} (use like C#min, Fmaj)")
    root = match.group(1) + match.group(2)
    root_pc = note_name_to_pc(root.replace("B", "b").replace("#", "#").upper().replace("b", "B"))
    return root_pc, match.group(3) == "MIN"


def default_profiles() -> Dict[str, StyleProfile]:
    trap_rhythm = [
        [(0.0, 2.0, 0.9), (2.0, 2.0, 0.85)],
        [(0.0, 1.0, 0.85), (1.0, 1.0, 0.75), (2.0, 2.0, 0.9)],
        [(0.0, 1.5, 0.85), (1.5, 0.5, 0.6), (2.0, 2.0, 0.9)],
    ]
    neo_rhythm = [
        [(0.0, 1.0, 0.75), (1.0, 1.0, 0.65), (2.0, 1.0, 0.75), (3.0, 1.0, 0.65)],
        [(0.0, 2.0, 0.8), (2.0, 1.0, 0.65), (3.0, 1.0, 0.65)],
    ]

    minor_trap = [
        [(1, "min9"), (4, "min11"), (7, "13"), (3, "maj9")],
        [(6, "maj7"), (7, "7alt"), (1, "min9"), (4, "min11")],
        [(1, "min9"), (5, "m7b5"), (1, "min9"), (7, "7alt")],
    ]
    major_pop = [
        [(1, "maj9"), (5, "7"), (6, "min7"), (4, "maj7")],
        [(1, "maj7"), (6, "min9"), (2, "min7"), (5, "7")],
    ]
    dark_minor = [
        [(1, "min7"), (7, "7alt"), (6, "maj7"), (5, "m7b5")],
        [(1, "min9"), (7, "7alt"), (3, "maj7"), (4, "min7")],
    ]

    return {
        "trap_melodic": StyleProfile(
            name="trap_melodic",
            bpm_range=(120, 150),
            key_pool=["C#min", "F#min", "G#min", "D#min", "A#min"],
            progression_pool=minor_trap,
            rhythm_pool=trap_rhythm,
            chord_octave=(3, 4),
            bass_octave=(1, 2),
            spread_weights={"close": 0.55, "spread": 0.45},
            drop2_prob=0.35,
            inversion_max=3,
            ext_velocity=(45, 62),
            core_velocity=(75, 95),
            humanize_ms=18,
            swing=0.12,
        ),
        "trap_dark": StyleProfile(
            name="trap_dark",
            bpm_range=(125, 160),
            key_pool=["Dmin", "Emin", "Fmin", "Gmin", "C#min"],
            progression_pool=dark_minor,
            rhythm_pool=trap_rhythm,
            chord_octave=(2, 4),
            bass_octave=(1, 2),
            spread_weights={"close": 0.7, "spread": 0.3},
            drop2_prob=0.25,
            inversion_max=2,
            ext_velocity=(40, 58),
            core_velocity=(78, 100),
            humanize_ms=14,
            swing=0.06,
        ),
        "pop_rnb": StyleProfile(
            name="pop_rnb",
            bpm_range=(85, 115),
            key_pool=["Cmaj", "Dmaj", "Emaj", "Fmaj", "Gmaj", "Amaj", "Bmaj"],
            progression_pool=major_pop,
            rhythm_pool=neo_rhythm,
            chord_octave=(3, 5),
            bass_octave=(1, 2),
            spread_weights={"close": 0.4, "spread": 0.6},
            drop2_prob=0.45,
            inversion_max=4,
            ext_velocity=(50, 68),
            core_velocity=(70, 92),
            humanize_ms=22,
            swing=0.18,
        ),
    }


ARTIST_MAP = {
    "gunna": "trap_melodic",
    "lilbaby": "trap_melodic",
    "future": "trap_melodic",
    "youngthug": "trap_melodic",
    "travis": "trap_melodic",
    "travis scott": "trap_melodic",
    "21savage": "trap_dark",
    "kodak": "trap_dark",
    "pop smoke": "trap_dark",
    "popsmoke": "trap_dark",
    "drake": "pop_rnb",
    "theweeknd": "pop_rnb",
    "sza": "pop_rnb",
}


def pick_profile(artist: str, profiles: Dict[str, StyleProfile]) -> StyleProfile:
    artist_norm = normalize_artist(artist)
    for key, profile_key in ARTIST_MAP.items():
        if normalize_artist(key) == artist_norm:
            return profiles[profile_key]

    for key, profile_key in ARTIST_MAP.items():
        key_norm = normalize_artist(key)
        if key_norm in artist_norm or artist_norm in key_norm:
            return profiles[profile_key]

    return profiles["trap_melodic"]


# ----------------------------
# Rhythm and swing
# ----------------------------


def apply_swing(time: float, beat_len: float, swing: float) -> float:
    if swing <= 0:
        return time
    eighth = beat_len / 2.0
    grid_idx = int(round(time / eighth))
    if grid_idx % 2 == 1:
        return time + swing * eighth
    return time


def humanize_time(time: float, ms: int, rng: random.Random) -> float:
    if ms <= 0:
        return time
    jitter = rng.uniform(-ms, ms) / 1000.0
    return max(0.0, time + jitter)


def humanize_velocity(velocity: int, amount: int, rng: random.Random) -> int:
    return int(clamp(velocity + rng.randint(-amount, amount), 1, 127))


# ----------------------------
# MIDI generation
# ----------------------------


def generate_one(
    profile: StyleProfile,
    seed: int,
    bars: int,
    make_bass: bool,
    make_topline: bool,
) -> pretty_midi.PrettyMIDI:
    rng = random.Random(seed)
    bpm = rng.randint(profile.bpm_range[0], profile.bpm_range[1])
    key = rng.choice(profile.key_pool)
    key_root_pc, is_minor = parse_key(key)

    progression = rng.choice(profile.progression_pool)
    rhythm = rng.choice(profile.rhythm_pool)

    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    chord_inst = pretty_midi.Instrument(program=0, name="Chords")
    bass_inst = pretty_midi.Instrument(program=33, name="Bass")
    top_inst = pretty_midi.Instrument(program=80, name="Topline")

    beat_len = 60.0 / bpm
    bar_len = 4 * beat_len

    spread = "close" if rng.random() < profile.spread_weights["close"] else "spread"

    for bar in range(bars):
        degree, quality = progression[bar % len(progression)]
        root_pc = degrees_to_pc(key_root_pc, is_minor, degree)
        pcs = build_chord_pcs(root_pc, quality)

        inversion = rng.randint(0, min(profile.inversion_max, max(0, len(pcs) - 1)))
        drop2 = rng.random() < profile.drop2_prob

        chord_octave = rng.randint(profile.chord_octave[0], profile.chord_octave[1])
        voiced = voice_chord(
            pcs=pcs,
            root_pc=root_pc,
            base_octave=chord_octave,
            spread=spread,
            inversion=inversion,
            drop2=drop2,
            keep_root=True,
        )

        for start_beat, duration_beat, accent in rhythm:
            start = bar * bar_len + start_beat * beat_len
            end = start + duration_beat * beat_len

            start = apply_swing(start, beat_len, profile.swing)
            end = apply_swing(end, beat_len, profile.swing)
            start = humanize_time(start, profile.humanize_ms, rng)
            end = humanize_time(end, profile.humanize_ms, rng)

            for idx, note in enumerate(voiced):
                if idx <= 2:
                    base_velocity = rng.randint(profile.core_velocity[0], profile.core_velocity[1])
                else:
                    base_velocity = rng.randint(profile.ext_velocity[0], profile.ext_velocity[1])
                base_velocity = int(base_velocity * (0.85 + 0.3 * accent))
                velocity = humanize_velocity(base_velocity, 6, rng)

                chord_inst.notes.append(
                    pretty_midi.Note(
                        velocity=int(clamp(velocity, 1, 127)),
                        pitch=int(note),
                        start=float(start),
                        end=float(end),
                    )
                )

        if make_bass:
            bass_octave = rng.randint(profile.bass_octave[0], profile.bass_octave[1])
            bass_note = midi_note(root_pc, bass_octave)
            bass_hits = [(0.0, 1.0), (2.0, 0.5)] if rng.random() < 0.6 else [(0.0, 2.0)]
            for start_beat, duration_beat in bass_hits:
                start = bar * bar_len + start_beat * beat_len
                end = start + duration_beat * beat_len
                start = humanize_time(apply_swing(start, beat_len, profile.swing * 0.5), profile.humanize_ms, rng)
                end = humanize_time(apply_swing(end, beat_len, profile.swing * 0.5), profile.humanize_ms, rng)
                velocity = humanize_velocity(rng.randint(78, 105), 8, rng)
                bass_inst.notes.append(
                    pretty_midi.Note(
                        velocity=int(clamp(velocity, 1, 127)),
                        pitch=int(bass_note),
                        start=float(start),
                        end=float(end),
                    )
                )

        if make_topline:
            chord_tones = sorted({pc % 12 for pc in pcs})
            top_octave = rng.randint(profile.chord_octave[1], profile.chord_octave[1] + 1)
            steps = rng.randint(2, 4)
            time = bar * bar_len
            for _ in range(steps):
                pc = rng.choice(chord_tones)
                pitch = midi_note(pc, top_octave)
                duration = rng.choice([0.5, 0.75, 1.0]) * beat_len
                start = humanize_time(apply_swing(time, beat_len, profile.swing), profile.humanize_ms, rng)
                end = humanize_time(apply_swing(time + duration, beat_len, profile.swing), profile.humanize_ms, rng)
                velocity = humanize_velocity(rng.randint(62, 92), 10, rng)
                top_inst.notes.append(
                    pretty_midi.Note(
                        velocity=int(clamp(velocity, 1, 127)),
                        pitch=int(pitch),
                        start=float(start),
                        end=float(end),
                    )
                )
                time += rng.choice([0.5, 1.0, 1.5]) * beat_len

    midi.instruments.append(chord_inst)
    if make_bass:
        midi.instruments.append(bass_inst)
    if make_topline:
        midi.instruments.append(top_inst)

    return midi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artist", type=str, required=True, help="Artist name (used to pick a style profile)")
    parser.add_argument("--out", type=str, default="out", help="Output folder")
    parser.add_argument("--n", type=int, default=20, help="How many MIDI files to generate")
    parser.add_argument("--bars", type=int, default=8, help="How many bars per file")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducibility")
    parser.add_argument("--bass", action="store_true", help="Add bass track")
    parser.add_argument("--topline", action="store_true", help="Add topline track")
    args = parser.parse_args()

    profiles = default_profiles()
    profile = pick_profile(args.artist, profiles)

    os.makedirs(args.out, exist_ok=True)

    print(f"[i] artist='{args.artist}' -> profile='{profile.name}'")
    print(f"[i] generating n={args.n}, bars={args.bars}, out='{args.out}'")

    for idx in range(args.n):
        seed = args.seed + idx * 10007
        midi = generate_one(
            profile=profile,
            seed=seed,
            bars=args.bars,
            make_bass=args.bass,
            make_topline=args.topline,
        )
        safe_artist = normalize_artist(args.artist) or "artist"
        filename = os.path.join(args.out, f"{safe_artist}_{profile.name}_seed{seed}_{idx:03d}.mid")
        midi.write(filename)

    print("[✓] done.")


if __name__ == "__main__":
    main()
