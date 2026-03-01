import ffmpeg


for i in range (5) :
    ffmpeg.input(f"videos/videoplayback_{i+1}.mp4").output(f"audios/videoplayback_{i+1}.mp3").run()
    