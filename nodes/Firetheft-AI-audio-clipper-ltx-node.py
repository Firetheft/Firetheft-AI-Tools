import torch
from comfy_api.latest import io

class FiretheftAudioClipperLTX(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FiretheftAudioClipperLTX",
            display_name="Audio Clipper（LTX专用音频裁剪）",
            category="📜Firetheft AI Tools",
            description="Intelligently crops audio for LTX-Video. Outputs both the audio and the exact frame count needed for the video node.",
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
                io.Int.Output("actual_frames", display_name="actual_frames"),
            ]
        )

    @classmethod
    def execute(cls, audio, chunk_index, duration_nominal, frame_rate) -> io.NodeOutput:
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        # 1. Calculate the standard aligned duration for one chunk
        target_frames_per_chunk = duration_nominal * frame_rate
        n_nominal = round((target_frames_per_chunk - 1) / 8)
        frames_per_chunk = 1 + (8 * n_nominal)
        
        # 2. Calculate the seamless start point
        actual_start_frame = chunk_index * frames_per_chunk
        final_start_time = actual_start_frame / frame_rate
        
        # Theoretical duration
        final_duration_seconds = frames_per_chunk / frame_rate
        
        # 3. Perform cropping
        start_sample = int(final_start_time * sample_rate)
        num_samples_to_cut = int(final_duration_seconds * sample_rate)
        
        total_samples = waveform.shape[2]
        if start_sample >= total_samples:
            start_sample = max(0, total_samples - 1)
            
        end_sample = min(start_sample + num_samples_to_cut, total_samples)
        cropped_waveform = waveform[:, :, start_sample:end_sample]
        
        # 4. Calculate actual duration and LTX-aligned frames
        actual_duration_seconds = cropped_waveform.shape[2] / sample_rate
        
        # Convert back to frames and ensure 1+8n for LTX compatibility
        raw_frames = round(actual_duration_seconds * frame_rate)
        # We need the closest 1+8n that doesn't significantly exceed the audio
        n_actual = (raw_frames - 1) // 8
        if n_actual < 0: n_actual = 0
        
        # If we are at the very end of the file and it's just a tiny bit short of the next block,
        # we decide if we want to round up or down. Usually LTX nodes need a clean 1+8n.
        # Let's check if the remainder is significant.
        remainder = (raw_frames - 1) % 8
        if remainder >= 4: # Round up to next 1+8n if more than half block
            n_actual += 1
            
        actual_ltx_frames = 1 + (8 * n_actual)
        
        result_audio = {
            "waveform": cropped_waveform,
            "sample_rate": sample_rate
        }
        
        return io.NodeOutput(result_audio, final_start_time, actual_duration_seconds, actual_ltx_frames)

NODE_CLASS_MAPPINGS = {"FiretheftAudioClipperLTX": FiretheftAudioClipperLTX}
NODE_DISPLAY_NAME_MAPPINGS = {"FiretheftAudioClipperLTX": "Audio Clipper（LTX专用音频裁剪）"}
