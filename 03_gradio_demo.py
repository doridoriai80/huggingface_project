# 03_gradio_demo.py
import gradio as gr
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

def predict(text):
    result = classifier(text)
    label = result[0]["label"]
    score = round(result[0]["score"], 3)
    return f"{label} ({score})"

iface = gr.Interface(fn=predict, inputs="text", outputs="text", title="Sentiment Analysis Demo")
iface.launch()
