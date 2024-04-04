# --------------------------------------------------------
# SEEM -- Segment Everything Everywhere All At Once
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu), Jianwei Yang (jianwyan@microsoft.com)
# --------------------------------------------------------

import os
import warnings
import PIL
from PIL import Image
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import gradio as gr
import torch
import argparse
import whisper
import numpy as np
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
# target_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
target_dir = os.path.join(current_dir, 'utils')
print('target_dir: ', target_dir)
sys.path.append(target_dir)

from gradio import processing_utils
from xdecoder.BaseModel import BaseModel
from xdecoder import build_model
from .utils.distributed import init_distributed
from .utils.arguments import load_opt_from_config_files
from .utils.constants import COCO_PANOPTIC_CLASSES

from tasks import *

current_dir = os.path.dirname(os.path.abspath(__file__)) #in case app.py is run from a different working dir

def parse_option():
    parser = argparse.ArgumentParser('SEEM Demo', add_help=False)
    parser.add_argument('--conf_files', default=os.path.join(current_dir, "configs/seem/seem_focall_lang.yaml"), metavar="FILE", help='path to config file', )

    args, unknown_args = parser.parse_known_args()

    return args

'''
build args
'''
args = parse_option()
opt = load_opt_from_config_files(args.conf_files)
opt = init_distributed(opt)

# META DATA
cur_model = 'None'
if 'focalt' in args.conf_files:
    pretrained_pth = os.path.join('checkpoints/seem/', "seem_focalt_v2.pt")
    if not os.path.exists(pretrained_pth):
        os.system("wget {}".format("https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focalt_v2.pt"))
    cur_model = 'Focal-T'
elif 'focal' in args.conf_files:
    pretrained_pth = os.path.join('checkpoints/seem/', "seem_focall_v1.pt")
    if not os.path.exists(pretrained_pth):
        os.system("wget {}".format("https://hf-mirror.com/xdecoder/SEEM/resolve/main/seem_focall_v1.pt"))
    cur_model = 'Focal-L'


'''
build model
'''
model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)

'''
audio
'''
# audio = whisper.load_model("base",download_root=".")
audio = None
@torch.no_grad()
def inference(image, task, *args, **kwargs):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        if 'Video' in task:
            # print(image)
            return interactive_infer_video(model, audio, image, task, *args, **kwargs)
        else:
            # print(image)
            return interactive_infer_image(model, audio, image, task, *args, **kwargs)

class ImageMask(gr.components.Image):
    """
    Sets: source="canvas", tool="sketch"
    """

    is_template = True

    def __init__(self, **kwargs):
        super().__init__(source="upload", tool="sketch", interactive=True, **kwargs)

    def preprocess(self, x):
        if isinstance(x, str):
            x = {'image': x, 'mask': x}
        elif isinstance(x, dict):
            if (x['mask'] is None and x['image'] is None):
                x
            elif (x['image'] is None):
                x['image'] = str(x['mask'])
            elif (x['mask'] is None):
                x['mask'] = str(x['image']) #not sure why mask/mask is None sometimes, this prevents preprocess crashing
        elif x is not None:
            assert False, 'Unexpected type {0} in ImageMask preprocess()'.format(type(x))

        return super().preprocess(x)

class Video(gr.components.Video):
    """
    Sets: source="canvas", tool="sketch"
    """

    is_template = True

    def __init__(self, **kwargs):
        super().__init__(source="upload", **kwargs)

    def preprocess(self, x):
        return super().preprocess(x)

if __name__ == "__main__":
    '''
    launch app
    '''
    title = "SEEM: Segment Everything Everywhere All At Once"
    description = """
    <div style="text-align: center; font-weight: bold;">
        <span style="font-size: 18px" id="paper-info">
            [<a href="https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once" target="_blank">GitHub</a>]
            [<a href="https://arxiv.org/pdf/2304.06718.pdf" target="_blank">arXiv</a>]
        </span>
    </div>
    <div style="text-align: left; font-weight: bold;">
        <br>
        &#x1F32A Note: The current model is run on <span style="color:blue;">SEEM {}</span>, for <span style="color:blue;">best performance</span> refer to <a href="https://huggingface.co/spaces/xdecoder/SEEM" target="_blank"><span style="color:red;">our demo</span></a>.
        </p>
    </div>
    """.format(cur_model)

    '''Usage
    Instructions:
    &#x1F388 Try our default examples first (Sketch is not automatically drawed on input and example image);
    &#x1F388 For video demo, it takes about 30-60s to process, please refresh if you meet an error on uploading;
    &#x1F388 Upload an image/video (If you want to use referred region of another image please check "Example" and upload another image in referring image panel);
    &#x1F388 Select at least one type of prompt of your choice (If you want to use referred region of another image please check "Example");
    &#x1F388 Remember to provide the actual prompt for each promt type you select, otherwise you will meet an error (e.g., rember to draw on the referring image);
    &#x1F388 Our model by default support the vocabulary of COCO 133 categories, others will be classified to 'others' or misclassifed.
    '''

    article = "The Demo is Run on SEEM-Tiny."
    inputs = [
        ImageMask(label="[Stroke] Draw on Image", type="pil"), 
        gr.inputs.CheckboxGroup(choices=["Stroke", "Example", "Text", "Audio", "Video", "Panoptic"], type="value", label="Interative Mode"), 
        ImageMask(label="[Example] Draw on Referring Image", type="pil"), 
        gr.Textbox(label="[Text] Referring Text"), 
        gr.Audio(label="[Audio] Referring Audio", source="microphone", type="filepath"), 
        gr.Video(label="[Video] Referring Video Segmentation",format="mp4",interactive=True)
    ]
    interface = gr.Interface(
        fn=inference,
        inputs=inputs,
        outputs=[
            gr.Image(
    #        type="pil",
            height=240,
            tool="compose",
            label="Segmentation Results (COCO classes as label)"),
            gr.Video(
            label="Video Segmentation Results (COCO classes as label)", format="mp4"
            ),
        ],
        examples=[
        ["examples/corgi1.webp", ["Text"], "examples/corgi2.jpg", "The corgi.", None, None],
        ["examples/river1.png", ["Text", "Audio"], "examples/river2.png", "The green trees.", "examples/river1.wav", None],
        ["examples/zebras1.jpg", ["Example"], "examples/zebras2.jpg", "", None, None],
        ["examples/fries1.png", ["Example"], "examples/fries2.png", "", None, None],
        ["examples/placeholder.png", ["Video"], "examples/ref_vase.JPG", "", None, "examples/vasedeck.mp4"],
        ],
        title=title,
        description=description,
        article=article,
        allow_flagging='never',
        cache_examples=False,
        )

    interface.launch()
