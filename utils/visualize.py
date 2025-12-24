import cv2
import os
from typing import Tuple, Optional


def draw_predictions(
    image_path: str,
    label: int,
    fake_prob: float,
    save_path: Optional[str] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
):
    """
    Draw prediction results on an image.

    Args:
        image_path: Path to input image
        label: 0 = REAL, 1 = FAKE
        fake_prob: probability of FAKE
        save_path: optional output path
        bbox: optional face bounding box (x1, y1, x2, y2)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image {image_path}")

    text = f"{'FAKE' if label == 1 else 'REAL'} ({fake_prob:.2f})"
    color = (0, 0, 255) if label == 1 else (0, 255, 0)

    # Draw bounding box if provided
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Draw label background
    (w, h), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    cv2.rectangle(
        img,
        (10, 10),
        (10 + w + 6, 10 + h + 10),
        color,
        -1,
    )

    # Put text
    cv2.putText(
        img,
        text,
        (13, 10 + h + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)

    return img
