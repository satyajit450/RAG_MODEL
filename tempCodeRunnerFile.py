     chunks = []
            
            for segment in segments:
                chunks.append({
                    "title": segment.title,
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