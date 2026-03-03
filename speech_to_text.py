from faster_whisper import WhisperModel
import json

model = WhisperModel(
    "tiny",        
    device="cpu",
    compute_type="int8"   # VERY IMPORTANT → reduces RAM usage
)

segments, info = model.transcribe("audios/videoplayback_5.mp3")

chunks =[] 

for segment in segments:
    chunks.append({
        "Start": segment.start,   # dot notation
        "End": segment.end,       # dot notation
        "Text": segment.text      # dot notation
    })
for i in range(5):
    with open(f"chunks/video_{i+1}.json", "w") as f :
        json.dump(chunks , f)

