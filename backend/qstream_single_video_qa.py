#!/usr/bin/env python3
"""
Q-Stream single-video end-to-end QA demo.

Run selected video through:
1) pipeline_stage1.py
2) event_construction.py
3) event_description_ollama.py
4) question answering from events_with_descriptions.json

Example:
  python qstream_single_video_qa.py --video_path data/coin/videos/0R9pdc9dO3Q.mp4
data/coin/videos/Wh0Z5_wqNto.mp4
One-shot question:
  python qstream_single_video_qa.py --video_path data/coin/videos/0R9pdc9dO3Q.mp4 --question "What happens in the video?"
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


DEFAULT_MODEL = "qwen2.5vl:7b"
OLLAMA_HOST = "http://localhost:11434"


def run_command(command: List[str], env: Optional[Dict[str, str]] = None) -> None:
    print("\nRunning:")
    print(" ".join(command))

    process_env = os.environ.copy()
    if env:
        process_env.update(env)

    result = subprocess.run(command, env=process_env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}")


def check_ollama_server(host: str = OLLAMA_HOST) -> bool:
    try:
        response = requests.get(f"{host}/api/tags", timeout=5)
        response.raise_for_status()
        return True
    except Exception:
        return False


def start_ollama_if_needed(host: str = OLLAMA_HOST) -> None:
    if check_ollama_server(host):
        print("Ollama server is running.")
        return

    print("Ollama server is not running. Starting it now...")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=open("/tmp/ollama_qstream.log", "a"),
        stderr=subprocess.STDOUT,
    )

    for _ in range(20):
        time.sleep(1)
        if check_ollama_server(host):
            print("Ollama server started.")
            return

    raise RuntimeError("Ollama server failed to start. Try: ollama serve > /tmp/ollama.log 2>&1 &")


def ensure_model_exists(model: str) -> None:
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if model not in result.stdout and model.split(":")[0] not in result.stdout:
            print(f"Warning: model {model} not found. Run: ollama pull {model}")
        else:
            print(f"Ollama model found: {model}")
    except Exception as exc:
        print("Warning: could not check ollama model list:", exc)


def safe_video_id(video_path: Path) -> str:
    return video_path.stem


def make_sample_video(project_dir: Path, video_path: Path) -> None:
    sample_path = project_dir / "sample_video.mp4"

    if sample_path.exists() or sample_path.is_symlink():
        sample_path.unlink()

    try:
        sample_path.symlink_to(video_path.resolve())
    except Exception:
        shutil.copy2(video_path, sample_path)

    print("Prepared sample_video.mp4 for:", video_path)


def run_stage1_and_event_construction(project_dir: Path, video_path: Path, force: bool = False) -> Path:
    video_id = safe_video_id(video_path)
    output_root = project_dir / "outputs" / "coin" / video_id
    output_features = output_root / "output_features"
    output_events = output_root / "output_events"
    events_json = output_events / "events.json"

    if events_json.exists() and output_features.exists() and not force:
        print(f"4.2-4.4 already exists for {video_id}. Skipping.")
        return output_events

    print(f"\n=== Running 4.2-4.4 for {video_id} ===")

    make_sample_video(project_dir, video_path)

    for folder in ["output_features", "output_events"]:
        temp = project_dir / folder
        if temp.exists():
            shutil.rmtree(temp)

    run_command([sys.executable, "pipeline_stage1.py"], env={"HF_ENDPOINT": "https://hf-mirror.com"})

    if not (project_dir / "output_features").exists():
        raise RuntimeError("output_features was not created by pipeline_stage1.py")

    run_command([sys.executable, "event_construction.py"])

    if not (project_dir / "output_events").exists():
        raise RuntimeError("output_events was not created by event_construction.py")

    output_root.mkdir(parents=True, exist_ok=True)

    if output_features.exists():
        shutil.rmtree(output_features)
    if output_events.exists():
        shutil.rmtree(output_events)

    shutil.move(str(project_dir / "output_features"), str(output_features))
    shutil.move(str(project_dir / "output_events"), str(output_events))

    print("Saved processed outputs to:", output_root)
    return output_events


def run_event_description(project_dir: Path, video_path: Path, output_events_dir: Path, model: str, force: bool = False) -> Path:
    video_id = safe_video_id(video_path)

    events_path = output_events_dir / "events.json"
    desc_path = output_events_dir / "events_with_descriptions.json"

    if desc_path.exists() and not force:
        print(f"events_with_descriptions.json already exists for {video_id}. Skipping description.")
        return desc_path

    if not events_path.exists():
        raise FileNotFoundError(f"events.json not found: {events_path}")

    desc_script = project_dir / "event_description_ollama.py"
    if not desc_script.exists():
        raise FileNotFoundError("event_description_ollama.py not found in project folder.")

    frame_dir = project_dir / "temp_event_frames" / video_id

    print(f"\n=== Generating event descriptions for {video_id} ===")

    run_command([
        sys.executable,
        str(desc_script),
        "--video_path", str(video_path),
        "--events_path", str(events_path),
        "--output_path", str(desc_path),
        "--model", model,
        "--frame_dir", str(frame_dir),
    ])

    if not desc_path.exists():
        raise RuntimeError("events_with_descriptions.json was not created.")

    print("Saved:", desc_path)
    return desc_path


def load_events(desc_path: Path) -> List[Dict[str, Any]]:
    with open(desc_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    clean = []
    for e in events:
        start = float(e.get("start_time", 0.0))
        end = float(e.get("end_time", start))
        clean.append({
            "event_id": e.get("event_id"),
            "start_time": start,
            "end_time": end,
            "description": str(e.get("description", "")).strip(),
        })
    return clean


def build_evidence_context(events: List[Dict[str, Any]], max_events: int = 100) -> str:
    lines = []
    for e in events[:max_events]:
        lines.append(
            f"Event {e['event_id']} [{e['start_time']:.2f}s - {e['end_time']:.2f}s]: {e['description']}"
        )

    if len(events) > max_events:
        lines.append(f"... {len(events) - max_events} more events omitted due to context length.")

    return "\n".join(lines)


def ask_qwen_text(prompt: str, model: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1},
    }

    response = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=180)
    response.raise_for_status()
    return response.json().get("response", "").strip()


def answer_question(events: List[Dict[str, Any]], question: str, model: str) -> str:
    evidence = build_evidence_context(events)

    prompt = f"""
You are an evidence-grounded video question answering assistant.

Answer the user question using only the event descriptions below.
Each event has an event id and timestamp.

Rules:
- Use only the provided evidence.
- Cite relevant event ids and timestamps.
- If the evidence is not enough, say the event descriptions do not contain enough information.
- Do not invent details.
- Keep the answer concise.

Event evidence:
{evidence}

User question:
{question}

Answer:
""".strip()

    return ask_qwen_text(prompt, model)


def preview_events(events: List[Dict[str, Any]], n: int = 10) -> None:
    print("\nEvent preview:")
    for e in events[:n]:
        print(f"Event {e['event_id']} [{e['start_time']:.2f}s-{e['end_time']:.2f}s]: {e['description']}")
    if len(events) > n:
        print(f"... {len(events) - n} more events")


def interactive_loop(events: List[Dict[str, Any]], model: str) -> None:
    print("\nAsk questions about this video. Type 'exit' to stop.\n")
    while True:
        q = input("Question: ").strip()
        if q.lower() in {"exit", "quit", "q"}:
            break
        if not q:
            continue
        print("\nAnswer:")
        print(answer_question(events, q, model))
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", default="/root/autodl-tmp/Q_Stream")
    parser.add_argument("--video_path", default=None)
    parser.add_argument("--question", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--force", action="store_true", help="Rerun all processing.")
    parser.add_argument("--force_description", action="store_true", help="Rerun event description only.")
    parser.add_argument("--skip_processing", action="store_true", help="Use existing outputs/coin/{video_id}.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_dir = Path(args.project_dir).resolve()
    if not project_dir.exists():
        raise FileNotFoundError(f"Project dir not found: {project_dir}")

    os.chdir(project_dir)

    if args.video_path:
        video_path = Path(args.video_path)
    else:
        video_path = Path(input("Enter video path: ").strip())

    if not video_path.is_absolute():
        video_path = project_dir / video_path

    video_path = video_path.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print("Project dir:", project_dir)
    print("Video:", video_path)
    print("Model:", args.model)

    start_ollama_if_needed()
    ensure_model_exists(args.model)

    video_id = safe_video_id(video_path)
    output_events_dir = project_dir / "outputs" / "coin" / video_id / "output_events"

    if args.skip_processing:
        if not output_events_dir.exists():
            raise FileNotFoundError(f"Existing output_events folder not found: {output_events_dir}")
        print("Skipping 4.2-4.4. Using:", output_events_dir)
    else:
        output_events_dir = run_stage1_and_event_construction(project_dir, video_path, force=args.force)

    desc_path = run_event_description(
        project_dir=project_dir,
        video_path=video_path,
        output_events_dir=output_events_dir,
        model=args.model,
        force=args.force or args.force_description,
    )

    events = load_events(desc_path)
    print("\nLoaded events:", len(events))
    preview_events(events)

    if args.question:
        print("\nQuestion:")
        print(args.question)
        print("\nAnswer:")
        print(answer_question(events, args.question, args.model))
    else:
        interactive_loop(events, args.model)


if __name__ == "__main__":
    main()
