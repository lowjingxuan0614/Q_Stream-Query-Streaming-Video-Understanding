# 4.2 to 4.3
import cv2
import torch
import open_clip
from PIL import Image
import numpy as np
import os

#run from 4.2 to 4.3
def run_stage_1_to_3(video_path, output_dir, sample_fps=1):
    # auto choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[System Notice] Using device: {device}")

    #use pretrained CLIP model
    print("[4.3] Initializing CLIP encoder...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", # CLIP image encoder architecture
        pretrained="laion2b_s34b_b79k", # pretrained weights trained on LAION data
        device=device
    )

    # set the model to evaluation mode
    model.eval()

    # open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    # read the video metadata
    video_fps = cap.get(cv2.CAP_PROP_FPS) # Frames per second of the input video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Total number of frames in the video

    print("Success: Video read successfully")
    print(f"Video FPS: {video_fps:.2f}") #see frame per second of the video
    print(f"Total frames: {total_frames}") # total how many frames in the video

    #since sample fps is 1, we will sample one frame per second
    sample_interval = max(1, int(video_fps / sample_fps))

    frame_idx = 0
    sampled_embeddings = []
    sampled_timestamps = []

    print("\n[4.2] Starting streaming frame extraction...")

    while cap.isOpened():
        #read frame by frame, no access future
        #frame is image array: height x width x 3 (RGB channels)
        ret, frame = cap.read()

        # if read nothing, video ended, break the loop
        if not ret:
            break

        # sample frames at the specified interval
        if frame_idx % sample_interval == 0:
            current_time = frame_idx / video_fps
            # for example
            # frame_idx = 50
            # video_fps = 25
            # current_time = 2.0s
            print(f"Processing frame at {current_time:.2f} seconds")

            #convert the frame from bgr to rgb, open cv order
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #convert from numpy array to PIL image
            pil_img = Image.fromarray(rgb_frame)

            # Tensor needed for CLIP
            # resize image
            # center crop
            # convert image to tensot
            # normalize pixel value
            image_input = preprocess(pil_img).unsqueeze(0).to(device)

            #use CLIP to ENCODE the image into embedding  
            # no calculate gradients, only inference/encoding
            with torch.no_grad():
                frame_embedding = model.encode_image(image_input)
                frame_embedding = frame_embedding / frame_embedding.norm(dim=-1, keepdim=True) # normalize the embedding vector

            #save the embedding and timestamp
            sampled_embeddings.append(frame_embedding.cpu().numpy())
            sampled_timestamps.append(current_time)

            #eventually will ge something like this:
            #sampled_embeddings = [v1, v2, v3, ...]
            #sampled_timestamps = [0.0, 1.0, 2.0, ...]
           
        # go to the next frame   
        frame_idx += 1

    #release the video
    cap.release()

    if len(sampled_embeddings) == 0:
        print("Error: No frames were extracted successfully")
        return

    os.makedirs(output_dir, exist_ok=True)

    # convert to numpy array
    embeddings_array = np.vstack(sampled_embeddings)
    timestamps_array = np.array(sampled_timestamps)

    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings_array)
    np.save(os.path.join(output_dir, "timestamps.npy"), timestamps_array)

    print("\nStages 4.2 and 4.3 completed successfully")
    print(f"Generated embedding count: {len(embeddings_array)}")
    print(f"Embedding matrix shape: {embeddings_array.shape}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    TEST_VIDEO = "sample_video.mp4"
    OUTPUT_DATA_DIR = "./output_features"

    run_stage_1_to_3(TEST_VIDEO, OUTPUT_DATA_DIR, sample_fps=1)
