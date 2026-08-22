"""Listen to the 16 Stable Audio Open prompt candidates.

    uv run play-prompts 5
    uv run play-prompts 1 16
    uv run play-prompts all
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# ---------------------------------------------------------------------------
# The 16 prompts. Edit the strings; pick which one with: uv run play-prompts N
# ---------------------------------------------------------------------------

EUPHORIA = (
    "soaring euphoric synth arpeggios, shimmering bright pads, four-on-the-floor kick, "
    "rising ecstatic energy, C major, 120 BPM"
)
TRIUMPH = (
    "bold brass-like synth stabs, powerful driving drums, confident marching pulse, "
    "victorious and heroic, C major, 120 BPM"
)
PLAYFUL_JOY = (
    "bouncy plucked synths, light staccato marimba, skipping syncopated rhythm, "
    "cheerful and mischievous, C major, 120 BPM"
)
AWE = (
    "vast slow-swelling choral pads, distant shimmering bells, wide suspended chords, "
    "immense open space, breathtaking and sublime, C major, 120 BPM"
)
SERENITY = (
    "gentle sustained warm pad, soft sine tones, no percussion, slow breathing swells, "
    "deeply peaceful and still, C major, 120 BPM half-time"
)
TENDERNESS = (
    "intimate felt piano, soft analog warmth, close and delicate, small gentle gestures, "
    "loving and tender, C major, 120 BPM half-time"
)
CONTENTMENT = (
    "mellow rhodes chords, soft brushed drums, easy relaxed groove, unhurried and safe, "
    "quietly content, C major, 120 BPM"
)
REVERIE = (
    "blurred reversed pads, tape-saturated haze, weightless drifting texture, "
    "dreamlike and floating, C major, 120 BPM half-time"
)
MELANCHOLY = (
    "sparse minor piano, faint distant string swells, slow and restrained, "
    "quietly sad and withdrawn, A minor, 120 BPM half-time"
)
GRIEF = (
    "deep mournful cello drone, hollow low piano notes, heavy and slow, "
    "desolate and aching, A minor, 120 BPM half-time"
)
EMPTINESS = (
    "thin cold sine drone, long dead silences between notes, no warmth or movement, "
    "numb and hollow, A minor, 120 BPM half-time"
)
NOSTALGIA = (
    "distant detuned music box, worn tape wobble, faded and far away, "
    "bittersweet longing for something lost, A minor, 120 BPM half-time"
)
ANXIETY = (
    "restless ticking pulses, nervous tremolo strings, unstable jittering rhythm, "
    "uneasy and agitated, A minor, 120 BPM"
)
DREAD = (
    "low rumbling sub bass, creeping dissonant swells, slow inescapable approach, "
    "ominous and dreadful, A minor, 120 BPM"
)
ANGER = (
    "harsh distorted bass, aggressive pounding industrial drums, relentless and violent, "
    "furious, A minor, 120 BPM"
)
TENSION = (
    "sustained high dissonant strings, sharp irregular percussive hits, coiled and unresolved, "
    "suspenseful anticipation, A minor, 120 BPM"
)

PROMPTS: list[tuple[str, str]] = [
    ("euphoria", EUPHORIA),
    ("triumph", TRIUMPH),
    ("playful_joy", PLAYFUL_JOY),
    ("awe", AWE),
    ("serenity", SERENITY),
    ("tenderness", TENDERNESS),
    ("contentment", CONTENTMENT),
    ("reverie", REVERIE),
    ("melancholy", MELANCHOLY),
    ("grief", GRIEF),
    ("emptiness", EMPTINESS),
    ("nostalgia", NOSTALGIA),
    ("anxiety", ANXIETY),
    ("dread", DREAD),
    ("anger", ANGER),
    ("tension", TENSION),
]

DURATION_S = 12.0
STEPS = 100
SEED = 0
GUIDANCE = 7.0
NEGATIVE = "low quality, average quality"
PLAY_AUDIO = True
FORCE_REGEN = False

# On CUDA, prompts are generated together in one batched forward pass instead
# of one-at-a-time — a single 8s clip barely saturates a modern GPU, so
# batching is what actually uses the compute you're paying for. Override with
# PLAY_PROMPTS_BATCH=<n> if you hit VRAM limits or want to push it higher.
MAX_BATCH = max(1, int(os.environ.get("PLAY_PROMPTS_BATCH", "8")))

MODEL_ID = "stabilityai/stable-audio-open-1.0"
OUTPUT_DIR = Path.cwd() / "output" / "prompts"

_PROMPT_TO_INDEX = {text: i for i, (_, text) in enumerate(PROMPTS)}
_NAME_TO_INDEX = {name: i for i, (name, _) in enumerate(PROMPTS)}


def _clip_key(prompt: str) -> str:
    payload = f"{prompt}|{SEED}|{STEPS}|{DURATION_S}|{GUIDANCE}|{NEGATIVE}"
    return hashlib.sha1(payload.encode()).hexdigest()[:8]


def _wav_path(index: int, name: str, prompt: str) -> Path:
    return OUTPUT_DIR / f"{index + 1:02d}_{name}_{_clip_key(prompt)}.wav"


def _resolve_one(token: str | int) -> int:
    if isinstance(token, int) or (isinstance(token, str) and token.isdigit()):
        n = int(token)
        if not 1 <= n <= len(PROMPTS):
            raise SystemExit(f"prompt index must be 1–{len(PROMPTS)}, got {n}")
        return n - 1
    if not isinstance(token, str):
        raise SystemExit(f"cannot resolve prompt token: {token!r}")
    key = token.strip().lower().replace(" ", "_").replace("-", "_")
    if key in _NAME_TO_INDEX:
        return _NAME_TO_INDEX[key]
    if token in _PROMPT_TO_INDEX:
        return _PROMPT_TO_INDEX[token]
    names = ", ".join(f"{i + 1}:{name}" for i, (name, _) in enumerate(PROMPTS))
    raise SystemExit(f"unknown prompt {token!r}. use all, 1–16, or one of: {names}")


def resolve_selection(tokens: list[str]) -> list[int]:
    if len(tokens) == 1 and tokens[0].strip().lower() == "all":
        return list(range(len(PROMPTS)))
    return [_resolve_one(item) for item in tokens]


def _pick_device() -> tuple[str, object]:
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def _load_pipeline(device: str, dtype: object):
    import torch
    from diffusers import StableAudioPipeline

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"loading {MODEL_ID} on {device} ({dtype})…", flush=True)
    try:
        pipe = StableAudioPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    except Exception as exc:
        msg = str(exc).lower()
        if any(s in msg for s in ("gated", "401", "restricted", "authorization", "403")):
            raise SystemExit(
                "Stable Audio Open is a gated Hugging Face model.\n"
                "1. Accept the license: https://huggingface.co/stabilityai/stable-audio-open-1.0\n"
                "2. Log in: huggingface-cli login\n"
                "   (or set HF_TOKEN in the environment)"
            ) from exc
        raise
    pipe = pipe.to(device)
    if device == "mps":
        pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=None)
    generator_device = "cpu" if device == "mps" else device
    return pipe, torch.Generator(generator_device)


def _to_playable(audio) -> "object":
    import numpy as np

    wave = audio.T.float().cpu().numpy()
    peak = float(np.max(np.abs(wave)))
    if peak > 0:
        wave = wave / peak * 0.9
    return wave.astype(np.float32)


def _generate(pipe, generator, prompt: str, device: str):
    import torch

    gen = generator.manual_seed(SEED)
    result = pipe(
        prompt,
        negative_prompt=NEGATIVE,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        audio_end_in_s=DURATION_S,
        num_waveforms_per_prompt=1,
        generator=gen,
    )
    audio = _to_playable(result.audios[0])
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    return audio


def _generate_batch(pipe, generator_device: object, prompts: list[str], device: str):
    import torch

    gens = [torch.Generator(generator_device).manual_seed(SEED) for _ in prompts]
    result = pipe(
        list(prompts),
        negative_prompt=[NEGATIVE] * len(prompts),
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        audio_end_in_s=DURATION_S,
        num_waveforms_per_prompt=1,
        generator=gens,
    )
    audios = [_to_playable(a) for a in result.audios]
    if device == "cuda":
        torch.cuda.empty_cache()
    return audios


def _save(path: Path, audio, sample_rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate)


def _play(audio, sample_rate: int, label: str) -> None:
    import sounddevice as sd

    print(f"playing {label}  (Ctrl+C to skip)", flush=True)
    try:
        sd.play(audio, sample_rate, blocking=True)
    except KeyboardInterrupt:
        sd.stop()
        print("skipped playback", flush=True)


def _print_menu() -> None:
    from rich.console import Console

    console = Console(highlight=False)
    console.print("[bold]prompts[/bold]  [dim]uv run play-prompts 1–16[/dim]")
    for i, (name, _) in enumerate(PROMPTS):
        console.print(f"  [bold]{i + 1:2d}[/bold]  {name}")


def _print_prompt(index: int, name: str, prompt: str, n: int, total: int) -> None:
    from rich.console import Console
    from rich.panel import Panel

    console = Console(highlight=False)
    title = f"{index + 1}  {name}"
    if total > 1:
        title = f"{n}/{total}  {title}"
    console.print()
    console.print(Panel(prompt, title=title, title_align="left"))


def _process_selection(selection: list[int], state: dict, play_audio: bool, force: bool) -> None:
    entries = []
    for n, index in enumerate(selection, start=1):
        name, prompt = PROMPTS[index]
        path = _wav_path(index, name, prompt)
        entries.append(
            {"n": n, "index": index, "name": name, "prompt": prompt, "path": path, "audio": None}
        )

    pending = [e for e in entries if force or not e["path"].exists()]

    if pending:
        if state["pipe"] is None:
            device, dtype = _pick_device()
            if device == "cpu":
                print(
                    "warning: no GPU found; CPU generation is extremely slow. "
                    "this is a listen-check, not the production loop.",
                    flush=True,
                )
            pipe, generator = _load_pipeline(device, dtype)
            state["pipe"] = pipe
            state["generator"] = generator
            state["device"] = device
            state["sample_rate"] = int(pipe.vae.sampling_rate)

        pipe, generator, device = state["pipe"], state["generator"], state["device"]

        if device == "cuda" and len(pending) > 1:
            for start in range(0, len(pending), MAX_BATCH):
                chunk = pending[start : start + MAX_BATCH]
                print(f"generating batch of {len(chunk)} on cuda…", flush=True)
                audios = _generate_batch(
                    pipe, generator.device, [e["prompt"] for e in chunk], device
                )
                for e, audio in zip(chunk, audios):
                    e["audio"] = audio
        else:
            for e in pending:
                e["audio"] = _generate(pipe, generator, e["prompt"], device)

    sample_rate = state["sample_rate"]
    for e in entries:
        _print_prompt(e["index"], e["name"], e["prompt"], e["n"], len(entries))

        if e["audio"] is None:
            import soundfile as sf

            e["audio"], sample_rate = sf.read(e["path"], always_2d=True, dtype="float32")
            print(f"cached {e['path']}", flush=True)
        else:
            _save(e["path"], e["audio"], sample_rate)
            print(f"wrote {e['path']}", flush=True)

        if play_audio:
            _play(e["audio"], sample_rate, e["path"].name)


def _repl(state: dict, play_audio: bool, force: bool) -> None:
    from rich.console import Console

    console = Console(highlight=False)
    console.print()
    console.print(
        "[dim]model loads on first prompt and stays resident — "
        "type a number/name, 'all', 'list', or 'q' to quit[/dim]"
    )
    while True:
        try:
            raw = console.input("\n[bold]> [/bold]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not raw:
            continue
        if raw.lower() in ("q", "quit", "exit"):
            break
        if raw.lower() in ("list", "l"):
            _print_menu()
            continue
        try:
            selection = resolve_selection(raw.split())
        except SystemExit as exc:
            console.print(f"[red]{exc}[/red]")
            continue
        _process_selection(selection, state, play_audio, force)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate and play a Stable Audio Open clip for one of the 16 emotion prompts.",
        usage="uv run play-prompts [-h] [--list] [--no-play] [--force] [--repl] 1-16",
    )
    parser.add_argument(
        "prompts",
        nargs="*",
        metavar="N",
        help='prompt number 1–16, a name, or "all"',
    )
    parser.add_argument("--list", action="store_true", help="print the 16 prompts and exit")
    parser.add_argument("--no-play", action="store_true", help="generate/save wavs, do not play")
    parser.add_argument("--force", action="store_true", help="regenerate even if a matching wav exists")
    parser.add_argument(
        "--repl",
        action="store_true",
        help="load the model once and keep prompting for more, instead of exiting after one",
    )
    args = parser.parse_args(argv)

    if args.list or (not args.prompts and not args.repl):
        _print_menu()
        if not args.prompts and not args.list and not args.repl:
            raise SystemExit("pass a prompt number, e.g. uv run play-prompts 5 (or uv run play-prompts --repl)")
        return

    play_audio = PLAY_AUDIO and not args.no_play
    force = FORCE_REGEN or args.force
    state = {"pipe": None, "generator": None, "device": "cpu", "sample_rate": 44100}

    if args.repl:
        if args.prompts:
            _process_selection(resolve_selection(args.prompts), state, play_audio, force)
        _repl(state, play_audio, force)
        return

    _process_selection(resolve_selection(args.prompts), state, play_audio, force)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
