import argparse
import base64
import json
import subprocess
from pathlib import Path

import cv2
import requests


def extract_frame(video_path, timestamp, output_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_id = int(timestamp * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    success, frame = cap.read()
    cap.release()

    if not success:
        return False

    cv2.imwrite(str(output_path), frame)
    return True


def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def describe_image_with_ollama(image_path, model="llava"):
    image_b64 = image_to_base64(image_path)

    prompt = (
        "Describe the main action in this image as one short event sentence. "
        "Focus on who is doing what and the visible action. "
        "Do not mention uncertainty unless necessary. "
        "Output only one sentence."
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False
    }

    response = requests.post(
        "http://localhost:11434/api/generate",
        json=payload,
        timeout=120
    )

    response.raise_for_status()
    return response.json()["response"].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--events_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model", default="llava")
    parser.add_argument("--frame_dir", default="temp_event_frames")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    events_path = Path(args.events_path)
    output_path = Path(args.output_path)
    frame_dir = Path(args.frame_dir)

    frame_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(events_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    enriched_events = []

    for event in events:
        event_id = event["event_id"]
        start_time = float(event["start_time"])
        end_time = float(event["end_time"])

        middle_time = (start_time + end_time) / 2.0
        frame_path = frame_dir / f"event_{event_id}.jpg"

        print(f"Describing event {event_id}: {start_time:.2f}s - {end_time:.2f}s")

        ok = extract_frame(video_path, middle_time, frame_path)

        if not ok:
            description = "No visual description available."
        else:
            try:
                description = describe_image_with_ollama(frame_path, model=args.model)
            except Exception as e:
                description = f"Description failed: {e}"

        new_event = dict(event)
        new_event["description"] = description
        new_event["description_frame_time"] = middle_time
        new_event["description_frame_path"] = str(frame_path)

        enriched_events.append(new_event)

        print("  ->", description)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched_events, f, indent=2, ensure_ascii=False)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()