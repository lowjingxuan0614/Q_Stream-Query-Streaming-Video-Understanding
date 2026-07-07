#4.5
import json
import os
import numpy as np


def compute_event_embedding(event, frame_embeddings):
    """
    For each event, compute one event-level embedding
    by averaging all frame embeddings inside that event.
    """
    frame_indices = event["frame_indices"]
    event_vectors = frame_embeddings[frame_indices]
    event_embedding = np.mean(event_vectors, axis=0)
    return event_embedding


def forecast_query_types(event, event_embedding):
    """
    4.5 Query Forecasting

    This prototype does not predict the exact future question.
    Instead, it predicts likely question categories:
    - Who
    - What
    - When
    - How

    Since this is a prototype, we use simple rules.
    """

    duration = event["end_time"] - event["start_time"]
    num_frames = event["num_frames"]

    likely_query_types = []
    reasons = []

    # Every event has timestamp evidence, so it can support When questions.
    likely_query_types.append("When")
    reasons.append("The event has a timestamp range, so it can answer timing or sequence questions.")

    # If the event contains visual frames, it can support What questions.
    if num_frames >= 1:
        likely_query_types.append("What")
        reasons.append("The event contains visual frame embeddings, so it can answer action/object/event questions.")

    # If the event lasts for multiple seconds, it may describe a procedure.
    if duration >= 2:
        likely_query_types.append("How")
        reasons.append("The event spans multiple seconds, so it may describe a procedural step or method.")

    # Who is included as a lower-confidence category.
    # At this stage we do not run person detection, so we cannot strongly confirm people.
    if num_frames >= 3:
        likely_query_types.append("Who")
        reasons.append("The event contains multiple frames, so it may include people or entities, but this is lower confidence.")

    # Remove duplicates while preserving order.
    seen = set()
    unique_query_types = []
    for q in likely_query_types:
        if q not in seen:
            unique_query_types.append(q)
            seen.add(q)

    return unique_query_types, reasons


def run_query_forecasting(
    events_path="output_events/events.json",
    embeddings_path="output_features/embeddings.npy",
    output_dir="output_query_forecasts"
):
    """
    Input:
    - events.json from 4.4
    - embeddings.npy from 4.3

    Output:
    - query_forecasts.json
    - query_forecasts.npy
    """

    with open(events_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    frame_embeddings = np.load(embeddings_path)

    forecasts = []

    print("Loaded events:", len(events))
    print("Loaded frame embeddings:", frame_embeddings.shape)
    print("\n【4.5】Started Query Forecasting...")

    for event in events:
        event_embedding = compute_event_embedding(event, frame_embeddings)
        likely_query_types, reasons = forecast_query_types(event, event_embedding)

        forecast = {
            "event_id": event["event_id"],
            "start_time": event["start_time"],
            "end_time": event["end_time"],
            "num_frames": event["num_frames"],
            "likely_query_types": likely_query_types,
            "reasons": reasons,
            "event_embedding_dim": int(event_embedding.shape[0])
        }

        forecasts.append(forecast)

        print(
            f"Event {event['event_id']}: "
            f"{event['start_time']:.2f}s - {event['end_time']:.2f}s "
            f"→ likely query types: {likely_query_types}"
        )

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "query_forecasts.json"), "w", encoding="utf-8") as f:
        json.dump(forecasts, f, indent=2, ensure_ascii=False)

    np.save(os.path.join(output_dir, "query_forecasts.npy"), np.array(forecasts, dtype=object))

    print("\n4.5 Query Forecasting completed successfully")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    run_query_forecasting()