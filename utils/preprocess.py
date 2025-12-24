import cv2
import argparse
from tqdm import tqdm
from pathlib import Path
from facenet_pytorch import MTCNN


def extract_frames_and_faces(
    input_dir: str,
    output_dir: str,
    frame_interval: int = 5,
    image_size: int = 224,
    max_frames: int = 40,
):
    """
    FINAL SAFE PREPROCESSING PIPELINE

    - Uses MTCNN ONLY for bounding box detection
    - Crops faces from original frames (no color artifacts)
    - Skips tiny / bad detections
    - Limits frames per video (prevents bias)
    - Logs videos with no detected faces

    CPU-only (Mac M2 safe)
    """

    # MTCNN ONLY for face localization
    detector = MTCNN(
        keep_all=False,
        device="cpu"   # MPS is unsafe for MTCNN
    )

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_counter = 0

    for label_dir in ["real", "fake"]:
        video_dir = input_dir / label_dir
        if not video_dir.exists():
            continue

        videos = sorted(video_dir.glob("*.mp4"))
        print(f"\n[INFO] Processing {label_dir.upper()} videos: {len(videos)}")

        for video_path in tqdm(videos):
            video_counter += 1
            video_id = f"{video_counter:04d}"

            save_dir = output_dir / video_id
            save_dir.mkdir(parents=True, exist_ok=True)

            cap = cv2.VideoCapture(str(video_path))
            frame_idx = 0
            saved = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    # Convert to RGB only for detection
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    boxes, _ = detector.detect(rgb)

                    if boxes is not None:
                        x1, y1, x2, y2 = boxes[0]
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                        # Clamp coordinates safely
                        h, w, _ = frame.shape
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        face = frame[y1:y2, x1:x2]

                        # 🔹 Improvement 1: skip tiny / bad detections
                        if face.size == 0 or face.shape[0] < 64 or face.shape[1] < 64:
                            continue

                        face = cv2.resize(face, (image_size, image_size))

                        out_path = save_dir / f"{video_id}_{saved:03d}.jpg"
                        cv2.imwrite(str(out_path), face)
                        saved += 1

                        # 🔹 Improvement 2: limit frames per video
                        if saved >= max_frames:
                            break

                frame_idx += 1

            cap.release()

            # 🔹 Improvement 3: log videos with no detected faces
            if saved == 0:
                save_dir.rmdir()
                print(f"[WARN] No face detected in {video_path.name}")

    print("\n✅ Preprocessing completed successfully (CLEAN, BALANCED DATASET).")


# ----------------------------------------------------
# CLI
# ----------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Deepfake preprocessing (FINAL IMPROVED)")

    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw_videos",
        help="Path to raw_videos directory",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/frames",
        help="Path to save extracted face frames",
    )

    parser.add_argument(
        "--frame_interval",
        type=int,
        default=5,
        help="Extract every Nth frame",
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Output face image size",
    )

    parser.add_argument(
        "--max_frames",
        type=int,
        default=40,
        help="Maximum frames saved per video",
    )

    args = parser.parse_args()

    extract_frames_and_faces(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        frame_interval=args.frame_interval,
        image_size=args.image_size,
        max_frames=args.max_frames,
    )
