from faster_whisper import WhisperModel

model = WhisperModel(
    "tiny",        # tiny is safest for your laptop
    device="cpu",
    compute_type="int8"   # VERY IMPORTANT → reduces RAM usage
)

segments, info = model.transcribe("audios/videoplayback_5.mp3")

for segment in segments:
    print(segment.text)