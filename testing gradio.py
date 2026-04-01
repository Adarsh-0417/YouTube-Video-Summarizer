import gradio as gr

def test(x):
    return "Working: " + x

gr.Interface(fn=test, inputs="text", outputs="text").launch()