import torch
from comfy_api.latest import io

class FiretheftAudioClipperLTX(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FiretheftAudioClipperLTX",
            display_name="Audio Clipper（LTX专用音频裁剪）",
            category="📜Firetheft AI Tools",
            description="Intelligently crops audio for LTX-Video. Use 'Chunk Index' to automatically skip to the correct seamless start point.",
            inputs=[
                io.Audio.Input("audio"),
                io.Int.Input("chunk_index", default=0, min=0, max=999, display_name="Chunk Index (Segment ID)"),
                io.Float.Input("duration_nominal", default=15.0, min=0.1, max=99999.0, step=0.01, display_name="Target Duration (s)"),
                io.Int.Input("frame_rate", default=24, min=1, max=120, display_name="Video FPS"),
            ],
            outputs=[
                io.Audio.Output("audio", display_name="audio"),
                io.Float.Output("actual_start", display_name="actual_start (s)"),
                io.Float.Output("actual_duration", display_name="actual_duration (s)"),
            ]
        )

    @classmethod
    def execute(cls, audio, chunk_index, duration_nominal, frame_rate) -> io.NodeOutput:
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        # 1. Calculate the standard aligned duration for one chunk
        # LTX requirement: Total Frames = 1 + 8*n
        target_frames_per_chunk = duration_nominal * frame_rate
        n = round((target_frames_per_chunk - 1) / 8)
        frames_per_chunk = 1 + (8 * n)
        
        # 2. Calculate the seamless start point based on chunk_index
        # Chunk 0 starts at frame 0
        # Chunk 1 starts at frame 361 (if chunk 0 was 361 frames)
        # Note: In a TRULY seamless non-overlapping audio sequence, 
        # Chunk 1 starts exactly where Chunk 0 ends.
        actual_start_frame = chunk_index * frames_per_chunk
        final_start_time = actual_start_frame / frame_rate
        
        # The duration remains the same aligned length
        final_duration = frames_per_chunk / frame_rate
        
        # 3. Perform cropping
        start_sample = int(final_start_time * sample_rate)
        num_samples = int(final_duration * sample_rate)
        
        total_samples = waveform.shape[2]
        if start_sample >= total_samples:
            # If index is out of bounds, return empty or last bit
            start_sample = max(0, total_samples - 1)
            
        end_sample = min(start_sample + num_samples, total_samples)
        cropped_waveform = waveform[:, :, start_sample:end_sample]
        
        actual_duration = cropped_waveform.shape[2] / sample_rate
        
        result_audio = {
            "waveform": cropped_waveform,
            "sample_rate": sample_rate
        }
        
        return io.NodeOutput(result_audio, final_start_time, actual_duration)

NODE_CLASS_MAPPINGS = {"FiretheftAudioClipperLTX": FiretheftAudioClipperLTX}
NODE_DISPLAY_NAME_MAPPINGS = {"FiretheftAudioClipperLTX": "Audio Clipper（LTX专用音频裁剪）"}
