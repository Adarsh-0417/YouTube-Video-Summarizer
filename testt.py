import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import torch
import gradio as gr
from transformers import pipeline

text_summary = None  # global variable

def get_model():
    global text_summary
    if text_summary is None:
        print("Loading model... (first time only)")
        text_summary = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            device=-1
        )
    return text_summary

# model_path = ("../Models/models--sshleifer--distilbart-cnn-12-6/snapshots"
#               "/a4f8f3ea906ed274767e9906dbaede7531d660ff")
# text_summary = pipeline("summarization", model=model_path,
#                 torch_dtype=torch.bfloat16)

def summary(input_text):
    model = get_model()

    max_chunk = 1000
    chunks = [input_text[i:i+max_chunk] for i in range(0, len(input_text), max_chunk)]
    
    summaries = []
    for chunk in chunks:
        out = model(chunk, max_length=150, min_length=40, do_sample=False)
        summaries.append(out[0]['summary_text'])
    
    return " ".join(summaries)

def extract_video_id(url):
    # Regex to extract the video ID from various YouTube URL formats
    regex = r"(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return None


def get_youtube_transcript(video_url):
    video_id = extract_video_id(video_url)
    if not video_id:
        return "Invalid URL"

    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)

        text_transcript = " ".join(
            str(getattr(item, "text", item)) for item in transcript
        )

        return summary(text_transcript)

    except Exception as e:
        return f"Error: {str(e)}"


# Example URL (Replace this with the actual URL when using the script)
# video_url = "https://youtu.be/5PibknhIsTc"
# print(get_youtube_transcript(video_url))

gr.close_all()

# demo = gr.Interface(fn=summary, inputs="text",outputs="text")
demo = gr.Interface(fn=get_youtube_transcript,
                    inputs=[gr.Textbox(label="Input YouTube Url to summarize",lines=1)],
                    outputs=[gr.Textbox(label="Summarized text",lines=4)],
                    title="@GenAILearniverse Project 2: YouTube Script Summarizer",
                    description="THIS APPLICATION WILL BE USED TO SUMMARIZE THE YOUTUBE VIDEO SCRIPT.")
print("Starting Gradio...")
demo.launch(share=True, debug=True)