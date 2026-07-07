#4.4
import numpy as np
import os
import json


def cosine_similarity(a, b):
    """
    Compute cosine similarity between two frame embeddings.
    Higher similarity means the two frames are more alike.
    Lower similarity means the visual content changed more.
    """
    #compare similarities with frame a and frame b
    #formula of cosine similarity: cos_sim = (a . b) / (||a|| * ||b||)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def construct_events(
    embeddings_path="output_features/embeddings.npy",
    timestamps_path="output_features/timestamps.npy",
    output_dir="output_events",
    similarity_threshold=0.75, #if similarity is below this threshold, we split the event
    max_event_length=5 # if event duration exceeds this length (in seconds), we split the event
):
    """
    4.4 Event Construction

    Input:
    - embeddings.npy: frame embeddings generated in 4.3
    - timestamps.npy: video time corresponding to each embedding

    Output:
    - events.json: start time, end time, and frame indices for each event
    - events.npy: the same event information for later Python use
    """

    embeddings = np.load(embeddings_path)
    timestamps = np.load(timestamps_path)

    print("Loaded embeddings:", embeddings.shape)
    print("Loaded timestamps:", timestamps)

    events = []

    current_start_idx = 0
    current_frame_indices = [0]

    print("\n[4.4] Starting event construction...")

    for i in range(1, len(embeddings)):
        sim = cosine_similarity(embeddings[i - 1], embeddings[i])
        event_duration = timestamps[i] - timestamps[current_start_idx]

        print(f"Frame {i-1} -> Frame {i} similarity: {sim:.4f}")

        should_split = False
        split_reason = None

        # Method 1: split the event when the embedding changes too much
        if sim < similarity_threshold:
            should_split = True
            split_reason = f"similarity below threshold ({sim:.4f} < {similarity_threshold})"

        # Method 2: use a fixed time window to avoid overly long events
        if event_duration >= max_event_length:
            should_split = True
            split_reason = f"max event length reached ({event_duration:.2f}s >= {max_event_length}s)"

        if should_split:
            #if we decide to split, we save the current event and start a new one
            event = {
                "event_id": len(events) + 1,
                "start_time": float(timestamps[current_start_idx]),
                "end_time": float(timestamps[i - 1]),
                "frame_indices": current_frame_indices,
                "num_frames": len(current_frame_indices),
                "split_reason": split_reason
            }
            events.append(event)

            current_start_idx = i
            current_frame_indices = [i]
        else:
            current_frame_indices.append(i)

    # Save the final event
    final_event = {
        "event_id": len(events) + 1,
        "start_time": float(timestamps[current_start_idx]),
        "end_time": float(timestamps[-1]),
        "frame_indices": current_frame_indices,
        "num_frames": len(current_frame_indices),
        "split_reason": "end of video"
    }
    events.append(final_event)

    os.makedirs(output_dir, exist_ok=True)

    # Save as npy for later Python use
    np.save(os.path.join(output_dir, "events.npy"), np.array(events, dtype=object))

    # Save as json for human review and reports
    with open(os.path.join(output_dir, "events.json"), "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)

    print("\n4.4 Event Construction completed successfully")
    print(f"Generated event count: {len(events)}")

    for e in events:
        print(
            f"Event {e['event_id']}: "
            f"{e['start_time']:.2f}s - {e['end_time']:.2f}s, "
            f"frames={e['frame_indices']}, "
            f"num_frames={e['num_frames']}"
        )


if __name__ == "__main__":
    construct_events()
