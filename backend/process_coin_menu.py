import os
import shutil
import subprocess
from pathlib import Path


VIDEO_DIR = Path("data/coin/videos")
CACHE_DIR = Path("outputs/coin")
SAMPLE_VIDEO = Path("sample_video.mp4")

OUTPUT_DIRS = [
    "output_features",
    "output_events",
    "output_query_forecasts",
    "output_memory",
    "output_answers",
]


def list_videos():
    videos = sorted(VIDEO_DIR.glob("*.mp4"))

    if not videos:
        print(f"No mp4 videos found in {VIDEO_DIR}")
        return []

    print("\nAvailable COIN videos:")
    print("=" * 60)

    for idx, video_path in enumerate(videos, start=1):
        video_id = video_path.stem
        size_mb = video_path.stat().st_size / (1024 * 1024)
        print(f"{idx}. {video_id}  ({size_mb:.2f} MB)")

    print("=" * 60)
    return videos


def choose_video(videos):
    while True:
        choice = input("\nChoose a video number to process: ").strip()

        if not choice.isdigit():
            print("Please enter a number.")
            continue

        choice = int(choice)

        if 1 <= choice <= len(videos):
            return videos[choice - 1]

        print("Invalid number. Try again.")


def clean_old_outputs():
    print("\nCleaning old output folders...")

    for folder in OUTPUT_DIRS:
        path = Path(folder)
        if path.exists():
            shutil.rmtree(path)
            print(f"Removed {folder}")


def copy_selected_video(video_path):
    print(f"\nSelected video: {video_path.name}")
    print(f"Copying to {SAMPLE_VIDEO} ...")

    shutil.copy2(video_path, SAMPLE_VIDEO)

    print("Done.")
    print(f"sample_video.mp4 size: {SAMPLE_VIDEO.stat().st_size / (1024 * 1024):.2f} MB")


def run_command(command, input_text=None):
    print("\nRunning:", " ".join(command))

    result = subprocess.run(
        command,
        input=input_text,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}")


def run_pipeline():
    print("\nStarting Q-Stream pipeline...")

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    # 4.2 + 4.3
    run_command(["python", "pipeline_stage1.py"])

    # 4.4
    run_command(["python", "event_construction.py"])

    # 4.5
    run_command(["python", "query_forecasting.py"])

    # 4.6
    run_command(["python", "memory_management.py"])

    # 4.7
    default_question = "What happened in the video?\n"
    run_command(["python", "answering.py"], input_text=default_question)

    print("\nPipeline completed.")


def save_outputs_to_cache(video_id):
    video_output_dir = CACHE_DIR / video_id
    video_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving outputs to {video_output_dir} ...")

    for folder in OUTPUT_DIRS:
        src = Path(folder)
        dst = video_output_dir / folder

        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"Saved {folder}")

    # also save a copy of selected video name
    metadata_path = video_output_dir / "selected_video.txt"
    metadata_path.write_text(f"{video_id}.mp4\n", encoding="utf-8")

    print("All outputs saved.")


def check_cached(video_id):
    memory_path = CACHE_DIR / video_id / "output_memory" / "memory.json"
    answer_path = CACHE_DIR / video_id / "output_answers" / "answer_result.json"

    return memory_path.exists() and answer_path.exists()


def main():
    # Change working directory to the project root 
    #os.chdir("D:\UOW\Degree\degree y3s3\IP2\Q_Stream") 

    videos = list_videos()
    if not videos:
        return

    selected_video = choose_video(videos)
    video_id = selected_video.stem

    cached = check_cached(video_id)

    if cached:
        print(f"\nCached results already exist for video_id: {video_id}")
        answer = input("Use cached result? Type y to skip processing, or n to reprocess: ").strip().lower()

        if answer == "y":
            print(f"\nCached result location: outputs/coin/{video_id}/")
            print("No processing needed.")
            return

    copy_selected_video(selected_video)
    clean_old_outputs()
    run_pipeline()
    save_outputs_to_cache(video_id)

    print("\nDone.")
    print(f"Processed video_id: {video_id}")
    print(f"Results saved at: outputs/coin/{video_id}/")


if __name__ == "__main__":
    main()