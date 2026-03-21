from faster_whisper import WhisperModel
import json
import os

model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)
audio_files = os.listdir("audios")
for audio in audio_files:
    if("_" in audio):
        title = audio.split(".")[0]
        if audio.endswith(".mp3"):
            segments, info = model.transcribe(f"audios/{audio}")

            chunks = []
            
            for segment in segments:
                chunks.append({
                    "Start": segment.start,
                    "End": segment.end,
                    "Text": segment.text
                })

            chunks_with_metadata = {
                "chunks": chunks,
                "full_text": " ".join([c["Text"] for c in chunks])
            }

            output_name = os.path.splitext(audio)[0] + ".json"

            with open(f"chunks/{output_name}", "w") as f:
                json.dump(chunks_with_metadata, f, indent=4)
            
       