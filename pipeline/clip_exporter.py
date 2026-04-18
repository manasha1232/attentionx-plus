"""
pipeline/clip_exporter.py
Stage 5: Vertical Clip Export

Uses MoviePy for video trimming and 9:16 vertical cropping.
Uses MediaPipe face detection for smart face-centered cropping.
Burns personalized captions as styled subtitles.
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


def get_face_crop_x(video_path: str, timestamp: float, video_width: int) -> int:
    """
    Use MediaPipe Face Detection to find the primary face's horizontal center.
    Returns x-offset for crop window. Falls back to center crop if no face found.
    """
    try:
        import mediapipe as mp
        import cv2

        mp_face = mp.solutions.face_detection
        cap = cv2.VideoCapture(video_path)

        # Seek to the target timestamp
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp * fps))

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return video_width // 2

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(rgb_frame)

            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                face_center_x = int((bbox.xmin + bbox.width / 2) * video_width)
                return face_center_x

    except Exception as e:
        print(f"[ClipExporter] MediaPipe face detection failed: {e}, using center crop")

    return video_width // 2


def create_caption_clip(
    text: str,
    duration: float,
    video_width: int,
    video_height: int,
) -> object:
    """Create a styled caption overlay using MoviePy TextClip."""
    try:
        from moviepy.editor import TextClip

        caption = (
            TextClip(
                text,
                fontsize=max(30, video_height // 18),
                font="DejaVu-Sans-Bold",
                color="white",
                stroke_color="black",
                stroke_width=2,
                method="caption",
                size=(int(video_width * 0.85), None),
                align="center",
            )
            .set_duration(duration)
            .set_position(("center", 0.75), relative=True)
        )
        return caption
    except Exception as e:
        print(f"[ClipExporter] Caption overlay failed: {e}")
        return None


def export_single_clip(
    video_path: str,
    start: float,
    end: float,
    caption_text: str,
    output_path: str,
    target_aspect: str = "9:16",
) -> str:
    """
    Export a single vertical clip with caption overlay.

    Args:
        video_path:    Source video
        start:         Clip start time (seconds)
        end:           Clip end time (seconds)
        caption_text:  Personalized caption to burn in
        output_path:   Output file path (.mp4)
        target_aspect: Output aspect ratio

    Returns:
        Path to exported clip
    """
    from moviepy.editor import VideoFileClip, CompositeVideoClip

    print(f"[ClipExporter] Exporting clip {start:.1f}s → {end:.1f}s")

    clip = VideoFileClip(video_path).subclip(start, end)
    orig_w, orig_h = clip.size

    # Determine crop dimensions for target aspect ratio
    if target_aspect == "9:16":
        target_w = int(orig_h * 9 / 16)
        target_h = orig_h

        if target_w > orig_w:
            # Video is already narrower than 9:16 — add letterbox via resize
            target_w = orig_w
            target_h = int(orig_w * 16 / 9)

    elif target_aspect == "1:1":
        side = min(orig_w, orig_h)
        target_w = target_h = side
    else:
        target_w, target_h = orig_w, orig_h

    # Smart face-centered horizontal crop
    face_x = get_face_crop_x(video_path, (start + end) / 2, orig_w)
    crop_x = max(0, min(face_x - target_w // 2, orig_w - target_w))

    # Crop to vertical
    cropped = clip.crop(x1=crop_x, y1=0, x2=crop_x + target_w, y2=target_h)

    # Add caption overlay
    caption_clip = create_caption_clip(caption_text, end - start, target_w, target_h)

    if caption_clip:
        final = CompositeVideoClip([cropped, caption_clip])
    else:
        final = cropped

    # Export
    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=30,
        preset="fast",
        verbose=False,
        logger=None,
    )

    clip.close()
    print(f"[ClipExporter] Saved: {output_path}")
    return output_path


def export_clips(
    video_path: str,
    highlights: List[Dict],
    output_dir: str,
    aspect_ratio: str = "9:16",
) -> List[str]:
    """
    Export all highlight clips with personalized captions.

    Args:
        video_path:   Source video
        highlights:   List of highlight dicts with start/end/personalized_caption
        output_dir:   Directory for output clips
        aspect_ratio: Target aspect ratio

    Returns:
        List of output file paths
    """
    output_paths = []
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for i, hl in enumerate(highlights):
        output_path = os.path.join(output_dir, f"clip_{i+1:02d}.mp4")
        caption = hl.get("personalized_caption", hl.get("text", ""))

        try:
            path = export_single_clip(
                video_path=video_path,
                start=hl["start"],
                end=hl["end"],
                caption_text=caption,
                output_path=output_path,
                target_aspect=aspect_ratio,
            )
            output_paths.append(path)
        except Exception as e:
            print(f"[ClipExporter] Failed clip {i+1}: {e}")
            output_paths.append(None)

    successful = [p for p in output_paths if p]
    print(f"[ClipExporter] Exported {len(successful)}/{len(highlights)} clips")
    return output_paths
