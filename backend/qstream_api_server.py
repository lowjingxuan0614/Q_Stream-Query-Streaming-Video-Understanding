#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_DIR = Path("/root/autodl-tmp/Q_Stream")
VIDEOS_DIR = PROJECT_DIR / "data/coin/videos"
COIN_JSON = PROJECT_DIR / "data/coin/COIN.json"
OUTPUTS_DIR = PROJECT_DIR / "outputs/coin"
OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5vl:7b"

JOBS: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="Q-Stream AI-only Backend")

# Allow local frontend at http://localhost:5500 to call this AutoDL backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve COIN videos through the backend.
# URL example: http://localhost:8000/coin_videos/xxx.mp4
if VIDEOS_DIR.exists():
    app.mount("/coin_videos", StaticFiles(directory=str(VIDEOS_DIR)), name="coin_videos")


class ProcessRequest(BaseModel):
    video_path: Optional[str] = None
    video_id: Optional[str] = None
    force: bool = False
    force_description: bool = False
    model: str = DEFAULT_MODEL


class AskRequest(BaseModel):
    video_id: str
    question: str
    model: str = DEFAULT_MODEL
    max_events: int = 120


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def current_time_for_log() -> str:
    return time.strftime("%I:%M:%S %p")

def clean_title(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.replace("_", " ")


def load_coin_title_map() -> Dict[str, str]:
    if not COIN_JSON.exists():
        return {}

    try:
        data = json.loads(COIN_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}

    records = data.get("database", data)
    if not isinstance(records, dict):
        return {}

    title_map: Dict[str, str] = {}

    for video_id, item in records.items():
        if not isinstance(item, dict):
            continue

        candidates: List[Any] = [
            item.get("class"),
            item.get("recipe_type"),
            item.get("category"),
            item.get("title"),
            item.get("name"),
            item.get("label"),
            item.get("task"),
        ]

        annotations = item.get("annotation")
        if isinstance(annotations, list) and annotations:
            first = annotations[0]
            if isinstance(first, dict):
                candidates.extend([
                    first.get("label"),
                    first.get("action"),
                    first.get("segment"),
                    first.get("name"),
                ])

        title = None
        for candidate in candidates:
            title = clean_title(candidate)
            if title:
                break

        if title:
            title_map[str(video_id)] = title

    return title_map


def add_job_log(job_id: str, message: str, log_type: str = "log-frame") -> None:
    if job_id not in JOBS:
        return

    JOBS[job_id].setdefault("logs", [])
    JOBS[job_id]["logs"].append({
        "time": current_time_for_log(),
        "message": message,
        "type": log_type,
    })

    if len(JOBS[job_id]["logs"]) > 1000:
        JOBS[job_id]["logs"] = JOBS[job_id]["logs"][-1000:]

    JOBS[job_id]["updated_at"] = now()


def parse_pipeline_stage1_line(job_id: str, line: str) -> None:
    line = line.strip()
    if not line:
        return

    if "[System Notice]" in line:
        add_job_log(job_id, line, "log-frame")
        return

    if "[4.3] Initializing CLIP encoder" in line:
        add_job_log(job_id, "Initializing pretrained OpenCLIP frame encoder", "log-encode")
        return

    if "Success: Video read successfully" in line:
        add_job_log(job_id, "Video read successfully", "log-frame")
        return

    if line.startswith("Video FPS:") or line.startswith("Total frames:"):
        add_job_log(job_id, line, "log-frame")
        return

    if "[4.2] Starting streaming frame extraction" in line:
        add_job_log(job_id, "Starting streaming frame sampling", "log-frame")
        return

    if line.startswith("Processing frame at"):
        JOBS[job_id]["frame_count"] = JOBS[job_id].get("frame_count", 0) + 1
        frame_no = JOBS[job_id]["frame_count"]

        match = re.search(r"at\s+([0-9.]+)\s+seconds", line)
        timestamp = match.group(1) if match else "?"

        add_job_log(job_id, f"Sampled frame #{frame_no} at {timestamp}s (fixed rate)", "log-frame")
        add_job_log(job_id, f"Encoded frame #{frame_no} → embedding: clip_emb_{frame_no:04d}", "log-encode")
        return

    if line.startswith("Generated embedding count:"):
        add_job_log(job_id, line, "log-encode")
        return

    if line.startswith("Embedding matrix shape:"):
        add_job_log(job_id, line, "log-encode")
        return

    if "Stages 4.2 and 4.3 completed successfully" in line:
        add_job_log(job_id, "Stages 4.2 and 4.3 completed successfully", "log-encode")
        return


def parse_event_construction_line(job_id: str, line: str) -> None:
    line = line.strip()
    if not line:
        return

    if line.startswith("Loaded embeddings:") or line.startswith("Loaded timestamps:"):
        add_job_log(job_id, line, "log-event")
        return

    if "[4.4] Starting event construction" in line:
        add_job_log(job_id, "Starting event construction from frame embeddings", "log-event")
        return

    if line.startswith("Event "):
        match = re.match(
            r"Event\s+(\d+):\s+([0-9.]+)s\s+-\s+([0-9.]+)s,\s+frames=(.*?),\s+num_frames=(\d+)",
            line,
        )
        if match:
            event_id, start, end, frames, num_frames = match.groups()
            add_job_log(
                job_id,
                f"New event detected: Event #{event_id} [{start}s → {end}s], frames={frames}, num_frames={num_frames}",
                "log-event",
            )
        else:
            add_job_log(job_id, line, "log-event")
        return

    if line.startswith("Generated event count:"):
        add_job_log(job_id, line, "log-event")
        return

    if "4.4 Event Construction completed successfully" in line:
        add_job_log(job_id, "4.4 Event Construction completed successfully", "log-event")
        return


def parse_description_line(job_id: str, line: str) -> None:
    line = line.strip()
    if not line:
        return

    lower = line.lower()
    if "event" in lower and ("description" in lower or "describing" in lower):
        add_job_log(job_id, line, "log-query")
    elif "saved" in lower or "completed" in lower or "done" in lower:
        add_job_log(job_id, line, "log-memory")


def run_cmd(
    job_id: str,
    cmd: List[str],
    env: Optional[Dict[str, str]] = None,
    parser: Optional[str] = None,
) -> None:
    print("\nRUN:", " ".join(cmd), flush=True)

    proc_env = os.environ.copy()
    proc_env["PYTHONUNBUFFERED"] = "1"
    if env:
        proc_env.update(env)

    process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_DIR),
        env=proc_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines: List[str] = []
    assert process.stdout is not None

    for line in process.stdout:
        clean = line.rstrip()
        output_lines.append(clean)
        print(clean, flush=True)

        if parser == "pipeline_stage1":
            parse_pipeline_stage1_line(job_id, clean)
        elif parser == "event_construction":
            parse_event_construction_line(job_id, clean)
        elif parser == "description":
            parse_description_line(job_id, clean)

    code = process.wait()
    if code != 0:
        output = "\n".join(output_lines[-80:])
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nLast output:\n{output}")


def check_ollama() -> bool:
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        response.raise_for_status()
        return True
    except Exception:
        return False


def start_ollama() -> None:
    if check_ollama():
        return

    subprocess.Popen(
        ["ollama", "serve"],
        stdout=open("/tmp/ollama_qstream_api.log", "a"),
        stderr=subprocess.STDOUT,
    )

    for _ in range(20):
        time.sleep(1)
        if check_ollama():
            return

    raise RuntimeError("Ollama server could not start. Try: ollama serve > /tmp/ollama.log 2>&1 &")


def resolve_video(video_path: Optional[str], video_id: Optional[str]) -> Path:
    if video_path:
        path = Path(video_path)
        if not path.is_absolute():
            path = PROJECT_DIR / path
    elif video_id:
        path = VIDEOS_DIR / f"{video_id}.mp4"
    else:
        raise HTTPException(status_code=400, detail="Provide video_path or video_id.")

    path = path.resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {path}")

    return path


def video_id_from_path(path: Path) -> str:
    return path.stem


def prepare_sample(video_path: Path) -> None:
    sample = PROJECT_DIR / "sample_video.mp4"
    if sample.exists() or sample.is_symlink():
        sample.unlink()

    try:
        sample.symlink_to(video_path.resolve())
    except Exception:
        shutil.copy2(video_path, sample)


def run_stage1_events(job_id: str, video_path: Path, force: bool = False) -> Path:
    video_id = video_id_from_path(video_path)
    out_root = OUTPUTS_DIR / video_id
    out_features = out_root / "output_features"
    out_events = out_root / "output_events"
    events_json = out_events / "events.json"

    if events_json.exists() and out_features.exists() and not force:
        add_job_log(job_id, "Existing 4.2-4.4 outputs found; skipping frame sampling, encoding, and event construction", "log-event")
        return out_events

    prepare_sample(video_path)

    for folder in ["output_features", "output_events"]:
        path = PROJECT_DIR / folder
        if path.exists():
            shutil.rmtree(path)

    run_cmd(
        job_id,
        [sys.executable, "-u", "pipeline_stage1.py"],
        env={"HF_ENDPOINT": "https://hf-mirror.com"},
        parser="pipeline_stage1",
    )

    if not (PROJECT_DIR / "output_features").exists():
        raise RuntimeError("output_features not created")

    run_cmd(
        job_id,
        [sys.executable, "-u", "event_construction.py"],
        parser="event_construction",
    )

    if not (PROJECT_DIR / "output_events").exists():
        raise RuntimeError("output_events not created")

    out_root.mkdir(parents=True, exist_ok=True)

    if out_features.exists():
        shutil.rmtree(out_features)
    if out_events.exists():
        shutil.rmtree(out_events)

    shutil.move(str(PROJECT_DIR / "output_features"), str(out_features))
    shutil.move(str(PROJECT_DIR / "output_events"), str(out_events))

    return out_events


def run_descriptions(job_id: str, video_path: Path, out_events: Path, model: str, force_description: bool = False) -> Path:
    video_id = video_id_from_path(video_path)
    events_path = out_events / "events.json"
    desc_path = out_events / "events_with_descriptions.json"

    if desc_path.exists() and not force_description:
        add_job_log(job_id, "Existing event descriptions found; skipping Qwen description generation", "log-query")
        return desc_path

    script = PROJECT_DIR / "event_description_ollama.py"
    if not script.exists():
        raise RuntimeError("event_description_ollama.py not found")

    frame_dir = PROJECT_DIR / "temp_event_frames" / video_id

    run_cmd(
        job_id,
        [
            sys.executable,
            "-u",
            str(script),
            "--video_path",
            str(video_path),
            "--events_path",
            str(events_path),
            "--output_path",
            str(desc_path),
            "--model",
            model,
            "--frame_dir",
            str(frame_dir),
        ],
        parser="description",
    )

    if not desc_path.exists():
        raise RuntimeError("events_with_descriptions.json not created")

    return desc_path


def load_events(video_id: str) -> List[Dict[str, Any]]:
    path = OUTPUTS_DIR / video_id / "output_events" / "events_with_descriptions.json"

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"events_with_descriptions.json not found for {video_id}. Run /process first.")

    return json.loads(path.read_text(encoding="utf-8"))


def evidence_text(events: List[Dict[str, Any]], max_events: int) -> str:
    lines = []
    for event in events[:max_events]:
        lines.append(
            f"Event {event['event_id']} "
            f"[{event['start_time']:.2f}s - {event['end_time']:.2f}s]: "
            f"{event.get('description', '')}"
        )

    if len(events) > max_events:
        lines.append(f"... {len(events) - max_events} more events omitted.")

    return "\n".join(lines)


def ask_qwen(prompt: str, model: str) -> str:
    start_ollama()

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.1,
            "num_predict": 350,
        },
    }

    response = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=180)
    response.raise_for_status()
    return response.json().get("response", "").strip()


def answer_video_question(video_id: str, question: str, model: str, max_events: int) -> Dict[str, Any]:
    events = load_events(video_id)
    evidence = evidence_text(events, max_events)

    prompt = f"""You are an evidence-grounded video question answering assistant.

Answer using only the event descriptions below.
Cite relevant event ids and timestamps.
If evidence is insufficient, say so.
Do not invent details.

Event evidence:
{evidence}

User question:
{question}

Answer:"""

    answer = ask_qwen(prompt, model)

    return {
        "video_id": video_id,
        "question": question,
        "answer": answer,
        "total_events": len(events),
        "num_events_used": min(len(events), max_events),
    }


def process_job(job_id: str, req: ProcessRequest) -> None:
    try:
        JOBS[job_id].update(status="running", message="Resolving video", updated_at=now())
        add_job_log(job_id, "Resolving selected COIN video", "log-frame")

        video_path = resolve_video(req.video_path, req.video_id)
        video_id = video_id_from_path(video_path)

        JOBS[job_id].update(
            video_id=video_id,
            video_path=str(video_path),
            message="Running 4.2-4.4",
            updated_at=now(),
        )

        add_job_log(job_id, f"Started processing: {video_path.name}", "log-frame")

        start_ollama()
        out_events = run_stage1_events(job_id, video_path, force=req.force)

        JOBS[job_id].update(message="Generating event descriptions", updated_at=now())
        add_job_log(job_id, "Generating event descriptions with Qwen", "log-query")
        desc_path = run_descriptions(
            job_id,
            video_path,
            out_events,
            req.model,
            force_description=req.force or req.force_description,
        )

        JOBS[job_id].update(
            status="done",
            message="Done",
            events_with_descriptions_path=str(desc_path),
            updated_at=now(),
        )
        add_job_log(job_id, "Done", "log-memory")
        add_job_log(job_id, f"Output: {desc_path}", "log-event")

    except Exception as exc:
        JOBS[job_id].update(status="failed", message=str(exc), updated_at=now())
        add_job_log(job_id, f"Backend processing failed: {exc}", "log-answer")


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "Q-Stream AI-only Backend",
        "message": "Run local frontend at http://localhost:5500 and call this backend through http://localhost:8000",
        "endpoints": ["/videos", "/coin_videos/{filename}", "/process", "/status/{job_id}", "/events/{video_id}", "/ask"],
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "name": "Q-Stream AI-only Backend",
        "videos": str(VIDEOS_DIR),
        "outputs": str(OUTPUTS_DIR),
    }


@app.get("/videos")
def videos() -> Dict[str, Any]:
    video_paths = sorted(VIDEOS_DIR.glob("*.mp4"))
    title_map = load_coin_title_map()

    items = []
    for path in video_paths:
        video_id = path.stem
        title = title_map.get(video_id)
        display_name = title or path.name

        items.append({
            "video_id": video_id,
            "filename": path.name,
            "title": title,
            "display_name": display_name,
            "path": str(path),
        })

    return {
        "count": len(items),
        "videos": items,
    }


@app.get("/video/{video_id}")
def video_file(video_id: str) -> FileResponse:
    video_path = VIDEOS_DIR / f"{video_id}.mp4"

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found.")

    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=video_path.name,
        headers={"Accept-Ranges": "bytes"},
    )


@app.post("/process")
def process(req: ProcessRequest) -> Dict[str, Any]:
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "message": "Queued",
        "logs": [],
        "frame_count": 0,
        "created_at": now(),
        "updated_at": now(),
    }

    thread = threading.Thread(target=process_job, args=(job_id, req), daemon=True)
    thread.start()

    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
def status(job_id: str) -> Dict[str, Any]:
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found.")

    return JOBS[job_id]


@app.get("/events/{video_id}")
def events(video_id: str) -> Dict[str, Any]:
    event_list = load_events(video_id)
    return {
        "video_id": video_id,
        "count": len(event_list),
        "events": event_list,
    }


@app.post("/ask")
def ask(req: AskRequest) -> Dict[str, Any]:
    return answer_video_question(req.video_id, req.question, req.model, req.max_events)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
