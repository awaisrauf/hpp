"""Container-level video metadata (passed to the LLM in the first prompt)."""

import cv2


def get_video_metadata(video_path: str) -> dict:
    """Get basic video metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Unable to open video"}

    metadata = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration_sec": None
    }

    if metadata["fps"] > 0:
        metadata["duration_sec"] = metadata["frame_count"] / metadata["fps"]
    metadata['duration'] = metadata['duration_sec']
    cap.release()
    return metadata
