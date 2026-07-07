#4.6 query memory management
#keep, merge, discard
import json
import os
import numpy as np


def calculate_importance_score(forecast):
    """
    4.6 Query-Aware Memory Management

    Calculate an importance score for each event based on
    the question types predicted in 4.5.
    A higher score means the event is more worth keeping.
    """

    query_types = forecast["likely_query_types"]
    num_frames = forecast["num_frames"]
    duration = forecast["end_time"] - forecast["start_time"]

    score = 0.0
    reasons = []

    # What is the most common video understanding question type, so it has a higher weight.
    if "What" in query_types:
        score += 0.30
        reasons.append("Contains What-type information about actions, objects, or events.")

    # How is important for instructional videos because COIN is a procedural video dataset.
    if "How" in query_types:
        score += 0.30
        reasons.append("Contains How-type procedural information.")

    # When supports timestamp-grounded answering.
    if "When" in query_types:
        score += 0.20
        reasons.append("Contains timestamp information for evidence-grounded answering.")

    # Who has lower confidence in this prototype because there is no person detector yet.
    if "Who" in query_types:
        score += 0.10
        reasons.append("May contain people or entities, but confidence is lower without person detection.")

    # Multi-frame events are more likely than single-frame events to contain a complete action.
    if num_frames >= 3:
        score += 0.10
        reasons.append("Contains multiple frames, which may represent a complete event.")

    # Very short events may contain less information.
    if duration <= 0:
        score -= 0.10
        reasons.append("Very short event, so it may contain limited context.")

    # Clamp the score between 0 and 1.
    score = max(0.0, min(score, 1.0))

    return score, reasons


def decide_memory_action(score):
    """
    Decide the memory action based on the importance score.
    """

    if score >= 0.75:
        return "keep"
    elif score >= 0.45:
        return "compress"
    else:
        return "forget"


def create_memory_summary(event, forecast, action, score):
    """
    Create a short memory summary.
    Since there is no caption model yet, the summary is generated
    from timestamps and query types for now.
    """

    time_range = f"{event['start_time']:.2f}s - {event['end_time']:.2f}s"
    query_types = ", ".join(forecast["likely_query_types"])

    if action == "keep":
        summary = (
            f"Event {event['event_id']} from {time_range} is kept in memory. "
            f"It may support {query_types} questions and contains {event['num_frames']} sampled frames."
        )
    elif action == "compress":
        summary = (
            f"Event {event['event_id']} from {time_range} is compressed. "
            f"Main query relevance: {query_types}."
        )
    else:
        summary = (
            f"Event {event['event_id']} from {time_range} is forgotten because it has low predicted usefulness."
        )

    return summary


def run_memory_management(
    events_path="output_events/events.json",
    forecasts_path="output_query_forecasts/query_forecasts.json",
    output_dir="output_memory"
):
    """
    Input:
    - events.json from 4.4
    - query_forecasts.json from 4.5

    Output:
    - memory.json
    - memory.npy
    """

    with open(events_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    with open(forecasts_path, "r", encoding="utf-8") as f:
        forecasts = json.load(f)

    forecast_by_id = {f["event_id"]: f for f in forecasts}

    memory_items = []

    print("Loaded events:", len(events))
    print("Loaded query forecasts:", len(forecasts))
    print("\n[4.6] Starting Query-Aware Memory Management...")

    for event in events:
        event_id = event["event_id"]
        forecast = forecast_by_id[event_id]

        score, score_reasons = calculate_importance_score(forecast)
        action = decide_memory_action(score)
        summary = create_memory_summary(event, forecast, action, score)

        memory_item = {
            "event_id": event_id,
            "start_time": event["start_time"],
            "end_time": event["end_time"],
            "num_frames": event["num_frames"],
            "likely_query_types": forecast["likely_query_types"],
            "importance_score": round(score, 3),
            "memory_action": action,
            "summary": summary,
            "score_reasons": score_reasons
        }

        memory_items.append(memory_item)

        print(
            f"Event {event_id}: "
            f"{event['start_time']:.2f}s - {event['end_time']:.2f}s | "
            f"score={score:.3f} | action={action}"
        )

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "memory.json"), "w", encoding="utf-8") as f:
        json.dump(memory_items, f, indent=2, ensure_ascii=False)

    np.save(os.path.join(output_dir, "memory.npy"), np.array(memory_items, dtype=object))

    print("\n4.6 Query-Aware Memory Management completed successfully")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    run_memory_management()
    
