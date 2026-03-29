import math
import random


class LTXLongAudioSegmentInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_duration": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 10000000.0, "step": 0.01}),
                "segment_seconds": ("INT", {"default": 20, "min": 1, "max": 3600, "step": 1}),
                "index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "INT", "FLOAT", "BOOLEAN", "INT")
    RETURN_NAMES = (
        "start_time",
        "segment_seconds",
        "frames",
        "exact_seconds",
        "is_last_segment",
        "segment_count",
    )
    FUNCTION = "segment_info"
    CATEGORY = "LTX/LongAudio"

    def segment_info(self, audio_duration, segment_seconds, index, fps):
        total_duration = max(float(audio_duration), 0.0)
        segment_seconds = max(int(segment_seconds), 1)
        index = max(int(index), 0)
        fps = max(float(fps), 1.0)

        segment_count = int(math.ceil(total_duration / segment_seconds)) if total_duration > 0 else 0
        start_time = float(index * segment_seconds)
        remaining = max(total_duration - start_time, 0.0)
        current_seconds = min(float(segment_seconds), remaining)
        is_last_segment = remaining < float(segment_seconds)

        frame_blocks = math.floor((current_seconds * fps) / 8.0) if current_seconds > 0 else 0
        frames = 1 + max(frame_blocks, 0) * 8
        exact_seconds = float((frames - 1) / fps) if frames > 1 else 0.0

        return (
            start_time,
            float(current_seconds),
            int(frames),
            exact_seconds,
            bool(is_last_segment),
            segment_count,
        )


class LTXRandomImageIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_count": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1}),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("image_index",)
    FUNCTION = "pick_index"
    CATEGORY = "LTX/LongAudio"

    def pick_index(self, image_count, segment_index, seed):
        image_count = max(int(image_count), 1)
        segment_index = max(int(segment_index), 0)
        seed = int(seed)

        rng = random.Random(seed + (segment_index * 9973))
        return (rng.randrange(image_count),)


NODE_CLASS_MAPPINGS = {
    "LTXLongAudioSegmentInfo": LTXLongAudioSegmentInfo,
    "LTXRandomImageIndex": LTXRandomImageIndex,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXLongAudioSegmentInfo": "LTX Long Audio Segment Info",
    "LTXRandomImageIndex": "LTX Random Image Index",
}
