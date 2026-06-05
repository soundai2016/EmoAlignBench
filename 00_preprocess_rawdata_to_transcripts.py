from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import glob
import json
import os
import ssl
import subprocess
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

import websockets

SUPPORTED_AUDIO = ("*.wav", "*.mp3", "*.mp4", "*.m4a", "*.flac", "*.aac", "*.ogg")


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    env_url = os.getenv("EMOALIGN_ASR_WS_URL")
    if env_url:
        cfg["asr_ws_url"] = env_url
    return cfg


def audio_bytes(wav_path: Path) -> bytes:
    if not wav_path.exists():
        return b""
    with wave.open(str(wav_path), "rb") as wf:
        return wf.readframes(wf.getnframes())


async def transcribe_wav(wav_path: Path, result_path: Path, ws_url: str, language: str) -> None:
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    async def send_audio(ws: Any) -> None:
        await ws.send(json.dumps({"signal": "start", "wav_name": wav_path.name, "language": language}))
        payload = audio_bytes(wav_path)
        for i in range(0, len(payload), 640):
            await ws.send(payload[i : i + 640])
            await asyncio.sleep(0.005)
        await ws.send(json.dumps({"signal": "end", "wav_name": wav_path.name}))

    async def receive(ws: Any) -> None:
        async for msg in ws:
            data = json.loads(msg)
            if data.get("mode") == "azero-full":
                data["timestamp"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                with open(result_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
            if data.get("is_final"):
                break

    async with websockets.connect(ws_url, subprotocols=["binary"], ssl=ssl_context) as ws:
        await asyncio.gather(send_audio(ws), receive(ws))


def to_wav_if_needed(audio_path: Path, sample_rate: int = 16000) -> Path:
    if audio_path.suffix.lower() == ".wav":
        return audio_path
    wav_path = audio_path.with_suffix(".wav")
    if wav_path.exists():
        return wav_path
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path), "-ar", str(sample_rate), "-ac", "1", "-c:a", "pcm_s16le", str(wav_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return wav_path


def process_file(audio_path: Path, cfg: Dict[str, Any]) -> None:
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"{audio_path.stem}.jsonl"
    if result_path.exists():
        print(f"skip existing transcript: {result_path.name}")
        return
    wav_path = to_wav_if_needed(audio_path)
    asyncio.run(transcribe_wav(wav_path, result_path, cfg["asr_ws_url"], cfg.get("language", "en")))
    print(f"wrote {result_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional ASR preprocessing for raw earnings-call audio.")
    parser.add_argument("--config", default="configs/asr_config.example.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_config(cfg_path)
    if "example-asr-endpoint" in cfg.get("asr_ws_url", ""):
        raise SystemExit("Set a real ASR websocket URL in a private config file or EMOALIGN_ASR_WS_URL.")

    audio_dir = Path(cfg["audio_dir"])
    if not audio_dir.is_absolute():
        audio_dir = root / audio_dir
    files = []
    for pattern in SUPPORTED_AUDIO:
        files.extend(Path(p) for p in glob.glob(str(audio_dir / "**" / pattern), recursive=True))
    print(f"found {len(files)} audio files")
    with ThreadPoolExecutor(max_workers=int(cfg.get("max_workers", 5))) as executor:
        futures = [executor.submit(process_file, f, cfg) for f in files]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
