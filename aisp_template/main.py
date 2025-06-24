#!/usr/bin/env python3
"""
pulse_transcribe.py – Record system audio via PulseAudio, slice into 5‑minute MP4
chunks, send each chunk asynchronously to OpenAI GPT‑4o‑transcribe, and
concatenate all transcripts into one file.

usage:
    python pulse_transcribe.py

Requirements:
    - Linux with PulseAudio / PipeWire (pactl available)
    - ffmpeg installed
    - pip install openai aiohttp aiofiles python-dateutil
    - OPENAI_API_KEY exported in environment
"""

import asyncio
import contextlib
import os
import shlex
import shutil
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import aiofiles
from openai import AsyncOpenAI

TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

CHUNK_SECONDS = 5 * 60  # 5 minutes
MODEL = "gpt-4o-transcribe"  # or "whisper-1"


# ---------------------------------------------------------------------------
# Step 1 – enumerate PulseAudio sources
# ---------------------------------------------------------------------------


def list_sources():
    """Return list[(index(str), name(str))] from pactl list short sources"""
    if shutil.which("pactl") is None:
        print("❌  pactl not found. Install PulseAudio utilities.")
        sys.exit(1)
    cmd = "pactl list short sources"
    output = subprocess.check_output(shlex.split(cmd), text=True)
    sources = []
    for line in output.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            index, name = parts[0], parts[1]
            sources.append((index, name))
    return sources


def choose_source():
    sources = list_sources()
    if not sources:
        print("❌  No PulseAudio sources found!")
        sys.exit(1)

    print("\nDostępne źródła (PulseAudio):")
    for i, (idx, name) in enumerate(sources):
        print(f"[{i}] {name} (index {idx})")

    while True:
        try:
            choice = int(input("\nWybierz źródło ▶ "))
            return sources[choice][1]
        except (ValueError, IndexError):
            print("Nieprawidłowy wybór – spróbuj ponownie.")


# ---------------------------------------------------------------------------
# Step 2 – record 5‑minute chunks using ffmpeg
# ---------------------------------------------------------------------------


def build_ffmpeg_cmd(source_name: str, outfile: Path):
    """Create ffmpeg command list to record PCM from PulseAudio to MP4"""
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "pulse",
        "-i",
        source_name,
        "-t",
        str(CHUNK_SECONDS),
        "-acodec",
        "aac",
        "-b:a",
        "128k",
        "-y",
        str(outfile),
    ]


# ---------------------------------------------------------------------------
# Step 3 – async transcription helper
# ---------------------------------------------------------------------------

client = AsyncOpenAI()


async def transcribe_file(path: Path) -> Path:
    """Send file to GPT‑4o‑transcribe and write .txt next to it; return txt path"""
    async with aiofiles.open(path, "rb") as audio_file:
        transcript_resp = await client.audio.transcriptions.create(
            model=MODEL,
            file=audio_file,
            response_format="text",
        )
    txt_path = path.with_suffix(".txt")
    async with aiofiles.open(txt_path, "w") as f:
        await f.write(transcript_resp)
    print(f"📝  Zapisano transkrypt {txt_path.name}")
    return txt_path


# ---------------------------------------------------------------------------
# Recording coroutine
# ---------------------------------------------------------------------------


async def recorder_loop(source_name: str, tasks: list[asyncio.Task]):
    """Loop forever: record CHUNK_SECONDS, then schedule transcription."""
    try:
        while True:
            start_ts = datetime.now(UTC)
            base = start_ts.strftime("%Y%m%dT%H%M%SZ")
            outfile = TEMP_DIR / f"{base}.mp4"
            cmd = build_ffmpeg_cmd(source_name, outfile)
            print(f"▶️  Nagrywam {outfile.name} … (ctrl‑C aby zakończyć)")
            proc = await asyncio.create_subprocess_exec(*cmd)
            await proc.wait()
            print(f"💾  Zakończono {outfile.name}")
            # schedule background transcription
            task = asyncio.create_task(transcribe_file(outfile))
            tasks.append(task)
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    source = choose_source()
    print(f"\nUżywam źródła: {source}\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("❌  Brak zmiennej środowiskowej OPENAI_API_KEY.")
        sys.exit(1)

    # Collect transcription tasks
    tasks: list[asyncio.Task] = []
    recorder = asyncio.create_task(recorder_loop(source, tasks))

    # graceful shutdown on ctrl‑C
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _sigint_handler():
        print("\n⏹️  Zatrzymywanie nagrywania…")
        recorder.cancel()
        stop_event.set()

    loop.add_signal_handler(signal.SIGINT, _sigint_handler)

    await stop_event.wait()

    # wait for recorder coroutine
    with contextlib.suppress(asyncio.CancelledError):
        await recorder

    print("⌛  Oczekiwanie na zakończenie transkrypcji…")
    txt_files: list[Path] = await asyncio.gather(*tasks, return_exceptions=False)
    txt_files.sort()

    # -------------------------------------------------------------------
    # Step 4 – concatenate transcripts into one final file
    # -------------------------------------------------------------------
    if txt_files:
        first = txt_files[0].stem
        last = txt_files[-1].stem
        final_path = TEMP_DIR / f"dialog_{first}_{last}.txt"
        async with aiofiles.open(final_path, "w") as fout:
            for t in txt_files:
                async with aiofiles.open(t) as fin:
                    await fout.write(f"----- {t.stem} -----\n")
                    await fout.write(await fin.read())
                    await fout.write("\n\n")
        print(f"✅  Gotowy plik: {final_path}")
    else:
        print("Brak transkrypcji do połączenia.")


if __name__ == "__main__":
    asyncio.run(main())
