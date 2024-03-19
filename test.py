#!/usr/bin/env python

import gradio as gr
from PIL import Image

def sendto(x):
    print(x)
    im = x["image"].copy()
    im.paste(x["mask"], (0, 0), x["mask"])
    return im

with gr.Blocks() as demo:
    with gr.Column():
        input_image = gr.Image(source='upload', type='pil', interactive=True, tool='sketch', elem_id="image_upload", brush_radius=200).style(height=400)
        ext_image = gr.Image(source="upload", type='pil', label="out", interactive=True).style(height=400)

    result_image = gr.Image(label="result", interactive=False).style(height=400)

    renderbtn = gr.Button("Render")
    renderbtn.click(sendto, inputs=[input_image], outputs=[result_image])

    sendtobtn = gr.Button("SendOutputToInput")
    sendtobtn.click(lambda x: x, inputs=[ext_image], outputs=[input_image])

demo.queue().launch(server_port=9876)