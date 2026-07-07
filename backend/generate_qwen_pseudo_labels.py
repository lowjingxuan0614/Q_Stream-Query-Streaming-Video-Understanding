#!/usr/bin/env python3
"""
Generate Qwen pseudo-labels for Q-Stream 4.5 and 4.6.

Input:
  outputs/coin/{video_id}/output_events/events_with_descriptions.json

Output folder:
  training/

Main output:
  training/qwen_pseudo_labels.json
  training/qwen_pseudo_labels.csv
  training/qwen_pseudo_label_summary.json

Each pseudo-label contains:
  - query_types_qwen: ["Who", "When", "What", "How"]
  - memory_action_qwen: "keep" / "compress" / "forget"
  - stored_memory_qwen: memory text or null
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


ALLOWED_QUERY_TYPES = ["Who", "When", "What", "How"]
ALLOWED_MEMORY_ACTIONS = ["keep", "compress", "forget"]


def call_ollama_text(
    model: str,
    prompt: str,
    host: str = "http://localhost:11434",
    timeout: int = 180,
    max_retries: int = 3,
    sleep_seconds: float = 1.0,
) -> str:
    """Call Ollama /api/generate with text-only prompt."""
    url = f"{host.rstrip('/')}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as exc:
            last_error = exc
            print(f"  Ollama call failed, attempt {attempt}/{max_retries}: {exc}")
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Ollama failed after {max_retries} retries: {last_error}")


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Extract a JSON object from the model output.
    This handles cases where the model accidentally adds extra text.
    """
    text = text.strip()

    # Remove markdown code fences if present.
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text}")

    return json.loads(match.group(0))


def clean_query_types(query_types: Any) -> List[str]:
    if not isinstance(query_types, list):
        return ["When", "What"]

    cleaned = []
    for item in query_types:
        if isinstance(item, str):
            item = item.strip()
            if item in ALLOWED_QUERY_TYPES and item not in cleaned:
                cleaned.append(item)

    if "When" not in cleaned:
        cleaned.append("When")

    if len(cleaned) == 0:
        cleaned = ["When", "What"]

    # Keep consistent order.
    return [q for q in ALLOWED_QUERY_TYPES if q in cleaned]


def clean_memory_action(action: Any) -> str:
    if not isinstance(action, str):
        return "compress"

    action = action.strip().lower()
    if action not in ALLOWED_MEMORY_ACTIONS:
        return "compress"

    return action


def build_prompt(
    video_id: str,
    event_id: int,
    start_time: float,
    end_time: float,
    duration: float,
    description: str,
) -> str:
    return f"""
You are a teacher model for a video question-answering memory system.

Given one event description and timestamp, generate pseudo-labels for training two student models:

4.5 Query Forecasting:
Predict which future question types this event can help answer.

Allowed query types:
- Who: questions about people, tools, objects, or entities involved
- When: questions about timing, order, or sequence
- What: questions about visible actions, objects, or events
- How: questions about procedures, methods, or step-by-step actions

4.6 Query-Aware Memory Management:
Decide how this event should be stored.

Allowed memory actions:
- keep: important, procedural, information-rich, or likely useful later
- compress: useful but simple, repetitive, or can be summarized shortly
- forget: unclear, redundant, or unlikely to help future question answering

Rules:
- Use only the given event description and timestamp.
- Do not invent visual facts that are not in the description.
- Include "When" if the event has a valid timestamp.
- Use "How" only if the description supports a method, procedure, operation, or step.
- stored_memory must be a short memory sentence with the timestamp if action is keep or compress.
- stored_memory must be null if action is forget.
- Return only valid JSON. Do not include markdown.

Event:
video_id: {video_id}
event_id: {event_id}
start_time: {start_time:.2f}
end_time: {end_time:.2f}
duration: {duration:.2f}
description: {description}

Return JSON exactly in this format:
{{
  "query_types": ["Who", "When", "What", "How"],
  "memory_action": "keep",
  "stored_memory": "0.00s-1.00s: short memory text"
}}
""".strip()


def fallback_label(start_time: float, end_time: float, description: str) -> Dict[str, Any]:
    """Safe fallback if Qwen fails."""
    if not description.strip():
        return {
            "query_types": ["When"],
            "memory_action": "forget",
            "stored_memory": None,
        }

    time_span = f"{start_time:.2f}s-{end_time:.2f}s"
    return {
        "query_types": ["When", "What"],
        "memory_action": "compress",
        "stored_memory": f"{time_span}: {description.strip()}",
    }


def load_events_with_descriptions(outputs_dir: Path) -> List[Dict[str, Any]]:
    description_files = sorted(outputs_dir.glob("*/output_events/events_with_descriptions.json"))

    records = []

    for path in description_files:
        video_id = path.parent.parent.parent.name

        with open(path, "r", encoding="utf-8") as f:
            events = json.load(f)

        for event in events:
            start_time = float(event.get("start_time", 0.0))
            end_time = float(event.get("end_time", start_time))
            duration = max(0.0, end_time - start_time)

            records.append({
                "video_id": video_id,
                "event_id": int(event.get("event_id", -1)),
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "num_frames": int(event.get("num_frames", 0)),
                "description": str(event.get("description", "")).strip(),
                "description_model": event.get("description_model", ""),
                "description_frame_time": event.get("description_frame_time", None),
                "description_frame_path": event.get("description_frame_path", ""),
                "source_file": str(path),
            })

    return records


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "video_id",
        "event_id",
        "start_time",
        "end_time",
        "duration",
        "num_frames",
        "description",
        "query_types_qwen",
        "memory_action_qwen",
        "stored_memory_qwen",
        "pseudo_label_model",
        "pseudo_label_status",
        "description_model",
        "description_frame_time",
        "description_frame_path",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            row = dict(record)
            row["query_types_qwen"] = json.dumps(row.get("query_types_qwen", []), ensure_ascii=False)
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", default="/root/autodl-tmp/Q_Stream")
    parser.add_argument("--model", default="qwen2.5vl:7b")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--limit", type=int, default=None, help="Optional small number for testing.")
    parser.add_argument("--resume", action="store_true", help="Skip labels already saved in training/qwen_pseudo_labels.json.")
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    outputs_dir = project_dir / "outputs/coin"
    training_dir = project_dir / "training"

    training_dir.mkdir(parents=True, exist_ok=True)

    output_json = training_dir / "qwen_pseudo_labels.json"
    output_csv = training_dir / "qwen_pseudo_labels.csv"
    summary_json = training_dir / "qwen_pseudo_label_summary.json"

    events = load_events_with_descriptions(outputs_dir)

    if args.limit is not None:
        events = events[:args.limit]

    print("Project dir:", project_dir)
    print("Events with descriptions found:", len(events))
    print("Training output folder:", training_dir)

    existing_by_key: Dict[str, Dict[str, Any]] = {}

    if args.resume and output_json.exists():
        with open(output_json, "r", encoding="utf-8") as f:
            existing_records = json.load(f)

        for record in existing_records:
            key = f"{record.get('video_id')}::{record.get('event_id')}"
            existing_by_key[key] = record

        print("Resume mode: existing labels:", len(existing_by_key))

    labeled_records = []
    query_counter = {}
    memory_counter = {}

    for index, event in enumerate(events, start=1):
        key = f"{event['video_id']}::{event['event_id']}"

        if args.resume and key in existing_by_key:
            labeled = existing_by_key[key]
            labeled_records.append(labeled)
            continue

        print(f"[{index}/{len(events)}] Labeling {event['video_id']} event {event['event_id']}")

        description = event["description"]
        start_time = event["start_time"]
        end_time = event["end_time"]

        try:
            prompt = build_prompt(
                video_id=event["video_id"],
                event_id=event["event_id"],
                start_time=start_time,
                end_time=end_time,
                duration=event["duration"],
                description=description,
            )

            raw_response = call_ollama_text(
                model=args.model,
                prompt=prompt,
                host=args.host,
            )

            parsed = extract_json_object(raw_response)

            query_types = clean_query_types(parsed.get("query_types"))
            memory_action = clean_memory_action(parsed.get("memory_action"))
            stored_memory = parsed.get("stored_memory")

            if memory_action == "forget":
                stored_memory = None
            elif not isinstance(stored_memory, str) or stored_memory.strip() == "":
                stored_memory = f"{start_time:.2f}s-{end_time:.2f}s: {description}"

            pseudo_label_status = "qwen_ok"

        except Exception as exc:
            print(f"  Failed to label event. Using fallback. Error: {exc}")
            fallback = fallback_label(start_time, end_time, description)
            query_types = fallback["query_types"]
            memory_action = fallback["memory_action"]
            stored_memory = fallback["stored_memory"]
            pseudo_label_status = f"fallback: {exc}"

        labeled = dict(event)
        labeled["query_types_qwen"] = query_types
        labeled["memory_action_qwen"] = memory_action
        labeled["stored_memory_qwen"] = stored_memory
        labeled["pseudo_label_model"] = args.model
        labeled["pseudo_label_status"] = pseudo_label_status

        labeled_records.append(labeled)

        for q in query_types:
            query_counter[q] = query_counter.get(q, 0) + 1
        memory_counter[memory_action] = memory_counter.get(memory_action, 0) + 1

        # Save progress after every event, so an interruption does not lose work.
        save_json(output_json, labeled_records)
        save_csv(output_csv, labeled_records)

    # Recompute counters including resumed labels.
    query_counter = {}
    memory_counter = {}

    for record in labeled_records:
        for q in record.get("query_types_qwen", []):
            query_counter[q] = query_counter.get(q, 0) + 1

        action = record.get("memory_action_qwen", "unknown")
        memory_counter[action] = memory_counter.get(action, 0) + 1

    summary = {
        "model": args.model,
        "total_labeled_events": len(labeled_records),
        "query_type_counts": query_counter,
        "memory_action_counts": memory_counter,
        "output_json": str(output_json),
        "output_csv": str(output_csv),
    }

    save_json(summary_json, summary)

    print("\nDone.")
    print("Saved:", output_json)
    print("Saved:", output_csv)
    print("Saved:", summary_json)
    print("Query type counts:", query_counter)
    print("Memory action counts:", memory_counter)


if __name__ == "__main__":
    main()
