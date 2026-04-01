import re
from youtube_transcript_api import YouTubeTranscriptApi
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Lazy load model
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        print("Loading model (first time only)...")

        tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
        model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")

    return tokenizer, model


def summarize_text(text):
    tokenizer, model = load_model()

    max_chunk = 800
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]

    summaries = []

    for chunk in chunks:
        if len(chunk.strip()) == 0:
            continue

        inputs = tokenizer(chunk, return_tensors="pt", truncation=True)

        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=120,
                min_length=30,
                num_beams=4
            )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    return " ".join(summaries)


def extract_video_id(url):
    regex = r"(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    return match.group(1) if match else None


def get_youtube_summary(video_url):
    video_id = extract_video_id(video_url)

    if not video_id:
        return "Invalid YouTube URL"

    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)

        text_parts = []
        for item in transcript:
            try:
                text_parts.append(str(item.text))
            except:
                text_parts.append(str(item))

        full_text = " ".join(text_parts)

        if len(full_text.strip()) == 0:
            return "Transcript is empty"

        return summarize_text(full_text)

    except Exception as e:
        return f"Error: {str(e)}"


# Gradio UI
demo = gr.Interface(
    fn=get_youtube_summary,
    inputs=gr.Textbox(label="Enter YouTube URL"),
    outputs=gr.Textbox(label="Summary"),
    title="YouTube Video Summarizer",
    description="Paste a YouTube link to get summary"
)

print("Starting Gradio...")
demo.launch(share=True)