import argparse
import base64
import io
import os
import sys
import re
import cv2
import gradio as gr
import numpy as np
import requests
from functools import partial
from PIL import Image, ImageOps
from app_utils import ImageBoxState, bbox_draw, open_image
import imageio
import tempfile
import Omegaconf
import torch
from diffusers import I2VGenXLPipeline
from diffusers.utils import export_to_gif, export_to_video
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from urllib.request import urlopen
from PIL import Image
from Vitron.vitron.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, OBJS_TOKEN_INDEX, DEFAULT_VIDEO_TOKEN
from Vitron.vitron.conversation import conv_templates, SeparatorStyle
from Vitron.vitron.model.builder import load_pretrained_model
from Vitron.vitron.utils import disable_torch_init
from Vitron.vitron.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, tokenizer_image_region_token, preprocess_region, show_image_with_bboxes

os.environ["BASE_HOME"] = "."

# sys.path.append(os.path.join(os.environ['BASE_HOME'], 'GLIGEN/demo'))
# import GLIGEN.demo.app as GLIGEN

# sys.path.append(os.path.join(os.environ['BASE_HOME'], 'SEEM/demo_code'))
# import SEEM.demo_code.app as SEEM  # must import GLIGEN_app before this. Otherwise, it will hit a protobuf error

# sys.path.append(os.path.join(os.environ['BASE_HOME'], 'Vitron'))

# sys.path.append(os.path.join(os.environ['BASE_HOME'], 'StableVideo'))
# import StableVideo.app as stablevideo


def save_image_to_local(image: Image.Image):
    # TODO: Update so the url path is used, to prevent repeat saving.
    if not os.path.exists('temp'):
        os.mkdir('temp')
    filename = os.path.join('temp', next(tempfile._get_candidate_names()) + '.jpg')
    image.save(filename)
    return filename


def save_video_to_local(video):
    if not os.path.exists('temp'):
        os.mkdir('temp')
    filename = os.path.join('temp', next(tempfile._get_candidate_names()) + '.mp4')
    writer = imageio.get_writer(filename, format='FFMPEG', fps=8)
    for frame in video:
        writer.append_data(frame)
    writer.close()
    return filename


def image_generation(prompt="a black swan swimming in a pond surrounded by green plants"):
    """
    :param prompt: text
    :return: 
    """
    sys.path.append(os.path.join(os.environ['BASE_HOME'], 'GLIGEN/demo'))
    import GLIGEN.demo.gligen.task_grounded_generation as GLIGEN
    cache_file = os.path.join('gligen/gligen-generation-text-box', 'diffusion_pytorch_model.bin')
    pretrained_ckpt_gligen = torch.load(cache_file, map_location='cpu')
    cache_config = os.path.join('gligen/demo_config_legacy', 'gligen-generation-text-box.pth')
    config = torch.load(cache_config, map_location='cpu')
    config = OmegaConf.create(config["_content"])
    config.update(
        {'folder': 'create_samples', 'official_ckpt': 'ckpts/sd-v1-4.ckpt', 'guidance_scale': 5, 'alpha_scale': 1})
    config.model['params']['is_inpaint'] = False
    config.model['params']['is_style'] = True
    loaded_model_list = GLIGEN.load_ckpt(config, pretrained_ckpt_gligen)
    # phrase_list = []
    # placeholder_image = Image.open('images/teddy.jpg').convert("RGB")
    # image_list = [placeholder_image] * len(phrase_list)
    instruction = dict(prompt=prompt, save_folder_name='gen_res', batch_size=1,
                       phrases=['placeholder'], has_text_mask=1, has_image_mask=0,
                       images=[], alpha_type=[0.3, 0, 0.7], guidance_scale=7.5, fix_seed=True,
                       rand_seed=0, actual_mask=None, inpainting_boxes_nodrop=None, locations=[])
    sample_list, over_list = GLIGEN.grounded_generation_box(loaded_model_list, instruction)
    image_path = save_image_to_local(sample_list[0])
    return image_path


def image_segmentation(image_path, track_text, sketch_pad=None):
    """
    Based on the input image, we segment the image and return the segmented image.
    Args:
        image (Image): The input image.
        reftxt (str): The reference text.
    Returns:
        Image: The segmented image.
    """
    print('Calling SEEM_app.inference')
    sys.path.append(os.path.join(os.environ['BASE_HOME'], 'SEEM/demo_code'))
    import SEEM.demo_code.app as SEEM
    import SEEM.demo_code.utils.visualizer as visual

    img = Image.open(image_path)
    width, height = img.size
    if len(track_text) == 0 and sketch_pad is None:
        # segment all
        compose_img = {'image': img, 'mask': img}
        task = []
        image, _, labels = SEEM.inference(image=compose_img, task=task, reftxt=track_text)
        return image[0], _, labels

    elif track_text:
        if sketch_pad is None:
            compose_img = {'image': img, 'mask': img}
        else:
            compose_img = sketch_pad
        task = ['Text']
        image, masks, labels = SEEM.inference(image=compose_img, task=task, reftxt=track_text)

        mask_pred = masks[0].astype("uint8")
        mask_pred = cv2.resize(mask_pred, (width, height), interpolation=cv2.INTER_LANCZOS4)
        mask_pred = mask_pred.astype("uint8")
        mask_demo = visual.GenericMask(mask_pred, height, width)
        bbox = mask_demo.bbox()
        mask = {'mask': mask_pred, 'boxes': bbox}
        return image[0], mask, labels

    elif sketch_pad:
        task = ['Stroke']
        image, expand, labels = SEEM.inference(image=sketch_pad, task=task, reftxt=track_text)
        return image[0], expand, labels

    return None


def image_editing(image_path="black-swan.png", sketch_pad=None,
               prompt="Turn the swan's neck into a wooden sail; Turn the swan into a wood boat"):
    """

    :param image_path: the path of the image that wu want to edit
    :param sketch_pad: the sketchpad input, formact: {"image": Image, "mask": Image}.
    :param prompt: text prompt
    :return: generate image with 512X512 resolution
    """
    # prompt="change the color of the swimming swan into blue"):
    sys.path.append(os.path.join(os.environ['BASE_HOME'], 'GLIGEN/demo'))
    import GLIGEN.demo.app as GLIGEN
    image = Image.open(image_path)
    width, height = image.size
    text = prompt
    texts = [x.strip() for x in text.split(';')]
    boxes = []
    masks = []
    for t in texts:
        _, t_mask, _ = image_segmentation(image_path=image_path, track_text=t)
        boxes.append(t_mask['boxes'])
        masks.append(t_mask['mask'])
    state = {'boxes': boxes}
    if sketch_pad is None:
        merged_mask = np.zeros((height, width))
        print(merged_mask.shape)
        for mask in masks:
            merged_mask = np.logical_or(merged_mask, mask)
        gen_images, state_list = GLIGEN.generate(task='Grounded Inpainting', language_instruction=prompt,
                                                 sketch_pad=None,
                                                 grounding_texts=prompt, alpha_sample=1.0, guidance_scale=30,
                                                 batch_size=1, fix_seed=False, rand_seed=0, use_actual_mask=False,
                                                 append_grounding=False,
                                                 style_cond_image=None, state=state, inpainting_mask=merged_mask,
                                                 inpainting_image=image
                                                 )
    else:
        gen_images, state_list = GLIGEN.generate(task='Grounded Inpainting', language_instruction=prompt,
                                                 sketch_pad=sketch_pad,
                                                 grounding_texts=prompt, alpha_sample=1.0, guidance_scale=30,
                                                 batch_size=1, fix_seed=False, rand_seed=0, use_actual_mask=False,
                                                 append_grounding=False,
                                                 style_cond_image=None, state=state, inpainting_mask=None,
                                                 inpainting_image=None
                                                 )

    # gen_images[0].save(save_path)
    image_path = save_image_to_local(gen_images[0])
    return image_path, state_list


def video_generation(prompt,
                   num_inference_steps=50,
                   num_frames=24,
                   guidance_scale=7.5,
                   ):
    """
    Based on the input text prompt, we generate the corresponding video.
    Args:
        prompt (str): The input text prompt.
        num_inference_steps (int): The number of inference steps.
        num_frames (int): The number of frames.
        guidance_scale (float): The guidance scale.
    """
    sys.path.append(os.path.join(os.environ['BASE_HOME'], 'Zeroscope'))
    pipe = DiffusionPipeline.from_pretrained("Zeroscope/zeroscope_v2_576w", torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    prompt = prompt
    video_frames = pipe(prompt, num_inference_steps=num_inference_steps, 
                        guidance_scale=guidance_scale,
                        height=320, width=576, 
                        num_frames=num_frames).frames
    video_path = save_video_to_local(video_frames)
    # video_path = export_to_video(video_frames, f"{save_dir}/video.mp4")
    return video_path


def video_tracking(video_path="vasedeck.mp4", sketchpad=None, track_prompt="", text_prompt=""):
    """
    Based on the input video, we track the video and return the tracked video.
    Args:
        video_path (str): The input video path.
        track_text (str): The track text.
        sketch_pad (dict): The sketchpad input with format {'image': Image, 'mask': Image}.
        track_prompt (str): The track prompt.
        text_prompt (str): The text prompt.
    Returns:
        str: The tracked video path.
    """
    sys.path.append(os.path.join(os.environ['BASE_HOME'], 'SEEM/demo_code'))
    import SEEM.demo_code.app as SEEM
    if sketchpad is None:
        i_video_path = video_path.split('/')[-2]
        img, o = video_editing(video_path=i_video_path, fore_prompt=text_prompt, back_prompt="")
        image_path = save_image_to_local(img)
        img = Image.open(image_path)
        compose_img = {'image': img, 'mask': img}
    else:
        compose_img = sketchpad

    _, output_video_name = SEEM.inference("SEEM/demo_code/examples/placeholder.png", task=['Video'],
                                          video_pth=video_path, refimg=compose_img, reftxt=track_prompt)
    return output_video_name


def video_editing(video_path="The_test_new_video.mp4", fore_prompt="turn the orange into bread",
               back_prompt="change the background into the blue"):
    """

    :param video_path (str): Path of the video to be modified
    :param fore_prompt (str): prompt for modifying foreground, such as "turn the orange into bread"
    :param back_prompt (str): prompt for modifying background, such as "change the background into the blue"
    """
    sys.path.append(os.path.join(os.environ['BASE_HOME'], 'StableVideo'))
    import StableVideo.app as stablevideo

    st = stablevideo.StableVideo(base_cfg="StableVideo/ckpt/cldm_v15.yaml",
                                 canny_model_cfg="StableVideo/ckpt/control_sd15_canny.pth",
                                 depth_model_cfg="StableVideo/ckpt/control_sd15_depth.pth",
                                 save_memory=False)

    st.load_canny_model(base_cfg='StableVideo/ckpt/cldm_v15.yaml',
                        canny_model_cfg='StableVideo/ckpt/control_sd15_canny.pth')
    st.load_depth_model(base_cfg='StableVideo/ckpt/cldm_v15.yaml',
                        depth_model_cfg='StableVideo/ckpt/control_sd15_depth.pth', )
    video_save_name, f_atlas_origin, b_altas_origin = st.load_video(video_name=video_path)
    print(video_save_name)

    f_atlas = st.advanced_edit_foreground(prompt=fore_prompt)
    # print(type(f_atlas))
    b_altas = st.edit_background(back_prompt)

    output_video_name = st.render_without_mask(f_atlas, b_altas)
    print(output_video_name)
    return f_atlas, output_video_name


def image_to_video(image_path='street.png', 
                   text_prompt='a car on the road, a white dog is beside the car, the dog run past the car',
                   ):
    """
    Based on the input image and text prompt, we generate the corresponding video.
    """
    sys.path.append(os.path.join(os.environ['BASE_HOME'], 'i2vgen-xl'))
    pipe = I2VGenXLPipeline.from_pretrained("i2vgen-xl/checkpoints", torch_dtype=torch.float16, variant="fp16")
    pipe.enable_model_cpu_offload()

    image = Image.open(image_path).convert("RGB")

    negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
    generator = torch.manual_seed(8800)

    frames = pipe(
        prompt=text_prompt,
        image=image,
        num_inference_steps=50,
        negative_prompt=negative_prompt,
        guidance_scale=9.0,
        generator=generator
    ).frames[0]
    # print(frames)
    # save 
    # for idx, img in enumerate(frames):
    #     img.save(f'{save_dir}/000{idx}.jpg')
    # video_path = export_to_gif(frames, f"{save_dir}/street.gif")
    # video_path = export_to_video(frames, f"{save_dir}/street_1.mp4")
    video_path = save_video_to_local(frames)
    return video_path
    # print(video_path)
  

def load_model(model_base, model_path, model_name='vitron-7b-lora'):
    disable_torch_init()
    cache_dir = 'cache_dir'
    device = 'cuda'
    load_4bit, load_8bit = False, False
    tokenizer, model, processor, _ = load_pretrained_model(model_path, model_base, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    image_processor = processor['image']
    video_processor = processor['video']
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    return tokenizer, model, image_processor, video_processor, conv


def find_module_content(data):
    pattern = r'<Module>(.*?)</Module>'
    match = re.search(pattern, data)
    if match:
        return match.group(1)
    else:
        return None


def find_instruction_content(data):
    pattern = r'<Instruction>(.*?)</Instruction>'
    match = re.search(pattern, data)
    if match:
        return match.group(1)
    else:
        return None

def find_region_instrction_content(data):
    pattern = r'<Region>(.*?)</Region>'
    match = re.search(pattern, data)
    if match:
        return match.group(1)
    else:
        return None


def remove_special_tags(text):
    pattern = r'<[^>]+>'  # match all the tags
    return re.sub(pattern, '', text)


def parse_model_output(model_output):
    """
    Based on the model output, we parse the model output and return the parsed instrcutions.
    Args:
        model_output (str): The model output.
    """
    # Parse the model output
    module = find_module_content(model_output)
    instruction = find_instruction_content(model_output)
    region = find_region_instrction_content(model_output)
    output = remove_special_tags(model_output)
    return output, module, instruction, region



# projection
tasks = {
    'A': image_generation,
    'B': image_segmentation,
    'C': image_editing,
    'D': video_generation,
    'E': video_tracking,
    'F': video_editing,
    'G': image_to_video,
}


def get_utterence(query, video_processor, image_processor):
    """
    Based on the query, we compose the corresponding utterence.
    Args:
        query (list): The input query. query[0]-> text, query[1]-> video, query[2]-> image
        video_processor (VideoProcessor): The video processor.  
        image_processor (ImageProcessor): The image processor.
    """
    res_utterance = ''
    video_tensor = None
    image_tensor = None
    if query[1] is not None and query[2] is not None:
        # input includes video and image
        res_utterance = DEFAULT_VIDEO_TOKEN + ' ' + DEFAULT_IMAGE_TOKEN + '\n' + query[0]
        video_tensor = video_processor(query[1], return_tensors='pt')['pixel_values']
        image_tensor = image_processor.preprocess(query[2], return_tensors='pt')['pixel_values']
    elif query[1] is not None and query[2] is None:
        # input includes video but no image
        res_utterance = DEFAULT_VIDEO_TOKEN + '\n' + query[0]
        video_tensor = video_processor(query[1], return_tensors='pt')['pixel_values']
    elif query[1] is None and query[2] is not None:
            # input includes no video but image
        res_utterance = DEFAULT_IMAGE_TOKEN + '\n' + query[0]
        video_tensor = video_processor(query[1], return_tensors='pt')['pixel_values']
    else:
        # input includes no video but image
        res_utterance = query[0]
    return res_utterance, video_tensor, image_tensor


def re_predict(tokenizer, model, image_processor, video_processor, conv,
             history, chatbox, user_input, 
             input_image_state, input_image, out_imagebox,
             input_video_state, input_video, video_sketch_pad,
            configs):
    q, a = history.pop()
    chatbox.pop()
    return predict(tokenizer, model, image_processor, video_processor, conv,
             history, chatbox, q, 
             input_image_state, input_image, out_imagebox,
             input_video_state, input_video, video_sketch_pad,
            configs)

def predict(tokenizer, model, image_processor, video_processor, conv,
             history, chatbox, user_input, 
             input_image_state, input_image, out_imagebox,
             input_video_state, input_video, video_sketch_pad,
            configs):
    """
    Based on the user input and history, we generate the response and update the history.
    """
    if history is not None:
        video_tensors = []
        image_tensors = []
        for idx, (q, a) in enumerate(history):
            q_utterance, q_video_tensor, q_image_tensor = get_utterence(q, video_processor, image_processor)
            conv.append_message(conv.roles[0], q_utterance)
            a_utterance, a_video_tensor, a_image_tensor = get_utterence(a, video_processor, image_processor)
            conv.append_message(conv.roles[1], a_utterance)
            if q_video_tensor is not None:
                video_tensors.append(q_video_tensor)
            if q_image_tensor is not None:
                image_tensors.append(q_image_tensor)
            if a_video_tensor is not None:
                video_tensors.append(a_video_tensor)
            if a_image_tensor is not None:
                image_tensors.append(a_image_tensor)
        
    inp = ''
    _user_input = ''
    if input_video_state.image is not None:
        inp += DEFAULT_VIDEO_TOKEN
        video_tensors.append(video_processor(input_video, return_tensors='pt')['pixel_values'])
        _user_input += f'<br><video controls playsinline width="500" style="display: inline-block;"  src="./file={input_video}"></video>'
    if input_image_state.image is not None:
        inp = inp + ' ' + DEFAULT_IMAGE_TOKEN
        image_tensors.append(image_processor.preprocess(input_image_state.image, return_tensors='pt')['pixel_values'])
        _user_input += f'<img src="./file={input_image_state.image}" style="display: inline-block;width: 250px;max-height: 400px;">'
    inp = inp + '\n' + user_input
    conv.append_message(conv.roles[0], inp)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=video_tensors+image_tensors,
            # regions = region,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    output, module, instruction, region = parse_model_output(outputs)
    
    if module and module in tasks:
        if module == 'A':
            image_path = image_generation(prompt=instruction)
            _response = output + f'<br><img src="./file={image_path}" style="display: inline-block;width: 250px;max-height: 400px;">'
            chatbox.append((_user_input, _response))
            history.append(((user_input, input_video, input_image_state.image), (output, None, image_path)))
        elif module == 'B':
            image_seg, pad, label = image_segmentation(image_path=input_image_state.image, track_text=instruction,
                                                        sketch_pad=out_imagebox)
            image_path = save_image_to_local(image_seg)
            _response = output + f'<br><img src="./file={image_path}" style="display: inline-block;width: 250px;max-height: 400px;">'   
            chatbox.append((_user_input, _response))    
            history.append(((user_input, input_video, input_image_state.image), (output, None, image_path)))
        elif module == 'C':
            image_path, _ = image_editing(image_path=input_image_state.image, sketch_pad=out_imagebox, prompt=instruction)
            _response = output + f'<br><img src="./file={image_path}" style="display: inline-block;width: 250px;max-height: 400px;">'
            chatbox.append((_user_input, _response))
            history.append(((user_input, input_video, input_image_state.image), (output, None, image_path)))
        elif module == 'D':
            video_path = video_generation(prompt=instruction)
            _response = output + f'<br><video controls playsinline width="500" style="display: inline-block;"  src="./file={video_path}"></video>'
            chatbox.append((_user_input, _response))
            history.append(((user_input, input_video, input_image_state.image), (output, video_path, None)))
        elif module == 'E':
            video_path = video_tracking(video_path=input_video, sketchpad=video_sketch_pad, track_prompt=instruction)
            _response = output + f'<br><video controls playsinline width="500" style="display: inline-block;"  src="./file={video_path}"></video>'
            chatbox.append((_user_input, _response))
            history.append(((user_input, input_video, input_image_state.image), (output, video_path, None)))
        elif module == 'F':
            _, video_path = video_editing(video_path=input_video, fore_prompt=instruction, back_prompt=instruction)
            _response = output + f'<br><video controls playsinline width="500" style="display: inline-block;"  src="./file={video_path}"></video>'
            chatbox.append((_user_input, _response))
            history.append(((user_input, input_video, input_image_state.image), (output, video_path, None)))
        elif module == 'G':
            video_path = image_to_video(image_path=input_image_state.image, text_prompt=instruction)
            _response = output + f'<br><video controls playsinline width="500" style="display: inline-block;"  src="./file={video_path}"></video>'
            chatbox.append((_user_input, _response))
            history.append(((user_input, input_video, input_image_state.image), (output, video_path, None)))
        else:
            raise NotImplementedError(f'The module {module} is not implemented.')

    return chatbox, history, None, None, None, None


default_chatbox = [("", "Please begin the chat.")]


def new_state():
    return {"ibs": ImageBoxState()}


def clear_fn2(value):
    return default_chatbox, None, new_state()


def reset_state(input_image_state, input_video_state):
    ibs = input_image_state["ibs"]
    ibs.reset_state()
    
    ibs = input_video_state["ibs"]
    ibs.reset_state()

    return None, None, None, None, [], [], []


def extract_frames(value, state: dict):
    """
    Based on the input video, we extract the video frames, and return the frames as a list of images.
    modified the code, as we do not need to extract all frames in the video, we only need to extract some frames.
    """
    # Get the video file path
    video_path = value
    # Open the video file   
    vidcap = cv2.VideoCapture(video_path)
    # Get the frame count
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get the frame rate
    frame_rate = int(vidcap.get(cv2.CAP_PROP_FPS))
    # Get the total duration of the video
    duration = frame_count / frame_rate
    # Get the number of frames to be extracted
    num_frames = 8
    # Get the frame interval
    frame_interval = int(frame_count / num_frames)
    # Get the frames
    frames = []
    # Loop through the frames
    for i in range(0, frame_count, frame_interval):
        # Set the frame position
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        # Read the frame
        success, frame = vidcap.read()
        # Convert the frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the frame to PIL image
        frame = Image.fromarray(frame)
        # Append the frame to the list of frames
        frames.append(frame)
    ibs = state["ibs"]
    ibs.update_image_list(frames)
    return frames[0], state


def edit_video_frame(sketch_pad, state): 
    """
    Based on the input video frame, we extract the video frame, and return the frame as an image.
    """
    def binarize(x):
        return (x != 0).astype('uint8') * 255
    image = sketch_pad['image']
    # print('sketch_pad', sketch_pad)  # {'image': array unit8, 'mask': array unit8}
    image = open_image(image)
    # global count
    # count += 1
    # np.save( f"{count}.npy", sketch_pad['mask'])
    mask = sketch_pad['mask'].sum(-1) if sketch_pad['mask'].ndim == 3 else sketch_pad['mask']
    mask = binarize(mask)
    ibs = state["ibs"]
    ibs.update_image(image)
    ibs.update_mask(mask)
    return image, state


def select_next_frame(state):
    ibs = state["ibs"]
    ibs.cnt += 1
    if ibs.cnt >= len(ibs.image_list):
        ibs.cnt = 0
    ibs.update_image(ibs.image_list[ibs.cnt])
    return ibs.image_list[ibs.cnt], state


def clear_video_and_frame(state):
    ibs = state["ibs"]
    # print(ibs.image_list)  # List[Image]
    # print(ibs.image)  # Image
    ibs.reset_state()

    return state, None, None


def clear_image_and_sketch_pad(state):
    ibs = state["ibs"]
    ibs.reset_state()

    return state, None, None


def clear_input(image_state, video_state):
    image_state["ibs"].reset_state()
    video_state["ibs"].reset_state()
    return image_state, video_state, None, None, None, None
    


TITLE = """
<h1 align="center" style=" display: flex; flex-direction: row; justify-content: center; font-size: 25pt; "><img src='./file=vitron.png' width="45" height="45" style="margin-right: 10px;">Vitron</h1>
<div align="center" style="display: flex;"><a href='https://vitron-llm.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp  &nbsp  &nbsp <a href='https://github.com/Vitron-LLM/Vitron'><img src='https://img.shields.io/badge/Github-Code-blue'></a> &nbsp &nbsp  &nbsp  <a href='https://arxiv.org'><img src='https://img.shields.io/badge/Paper-PDF-red'></a> &nbsp &nbsp  &nbsp  <a href='https://youtu.be/wiGMJzoQVu4'><img src='https://img.shields.io/badge/video-YouTube-FF0000'></a></div>
"""

INTRODUCTION = """
<h2>Introduction</h2>
<p>This is the demo page of Vitron, a universal pixel-level vision LLM, designed for comprehensive understanding (perceiving and reasoning), generating, segmenting (grounding and tracking), editing (inpainting) of both static image and dynamic video content.</p>
<h2>Term of Use</h2>
<p>The service is a research preview intended for non-commercial use only. The current initial version of Vitron, limited by the quantity of fine-tuning data and the quality of the base models, may generate some low-quality or hallucinated content. Please interpret the results with caution. We will continue to update the model to enhance its performance. Thank you for trying the demo! If you have any questions or feedback, feel free to contact us.</p>      
"""


def build_demo():
    demo = gr.Blocks(title="Vitron", css='style.css')
    with demo:
        gr.HTML(TITLE)
        gr.HTML(INTRODUCTION)

        with gr.Row():
            with gr.Column(scale=7, min_width=500):
                with gr.Row(): 
                    chatbot = gr.Chatbot(label='Vitron Chatbot', height=500, elem_id='chatbot', avatar_images=((os.path.join(os.path.dirname(__file__), 'vitron.png')), (os.path.join(os.path.dirname(__file__), "vitron.png"))))
                
                with gr.Row():
                    user_input = gr.Textbox(label='User Input', placeholder='Enter your text here', elem_id='user_input', lines=3)  
                
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Row():
                            input_image = gr.Image(label='Input Image', type='numpy',
                                                    shape=(512, 512), 
                                                    # height=200, width=200, 
                                                    elem_id='img2img_image', 
                                                    interactive=True, tool='sketch', 
                                                    brush_radius=20.0, visible=True)
                        with gr.Row():
                            clearImageBtn = gr.Button("Clear Image and Sketch Pad", elem_id='clear_image')
                    with gr.Column(scale=3):
                        with gr.Row():
                            input_video = gr.Video(label='Input Video', format='mp4', visible=True)  #.style(height=200) # , value=None, interactive=True
                        with gr.Row():
                            with gr.Column():
                                nextFrameBtn = gr.Button("Next Frame", elem_id='next_frmame', variant="primary")
                            with gr.Column(scale=1):
                                clearFrameBtn = gr.Button("Clear Video and Frame", elem_id='clear_frmame')
                
                with gr.Row():
                    with gr.Column(scale=3):
                        out_imagebox = gr.Image(label='Parsed Sketch Pad"', type='numpy',
                                                shape=(512, 512),  
                                                # height=200, width=200, 
                                                elem_id='img2img_image')
                        input_image_state = gr.State(new_state())
                    with gr.Column(scale=3):
                        video_sketch_pad = gr.Image(label='Video Frame', type='numpy', 
                                                    shape=(512, 512),
                                                    # height=200, width=200, 
                                                    elem_id='img2img_image', 
                                                    interactive=True, tool='sketch', 
                                                    brush_radius=20.0, visible=True)
                        input_video_state = gr.State(new_state())                     

            with gr.Column(scale=3, min_width=300):
                with gr.Group():
                    seed = gr.Slider(0, 9999, value=1234, label="SEED", interactive=True)
                    with gr.Accordion('Text Advanced Options', open=True):
                        top_p = gr.Slider(0, 1, value=0.01, step=0.01, label="Top P", interactive=True)
                        temperature = gr.Slider(0, 1, value=1.0, step=0.01, label="Temperature", interactive=True)
                    with gr.Accordion('Image Editing Advanced Options', open=True):
                        guidance_scale_for_img_edit = gr.Slider(1, 10, value=7.5, step=0.5, label="Guidance scale",
                                                        interactive=True)
                        num_inference_steps_for_img_edit = gr.Slider(10, 50, value=50, step=1, label="Number of inference steps",
                                                                interactive=True)
                    with gr.Accordion('Video Generation Options', open=False):
                        guidance_scale_for_vid = gr.Slider(1, 10, value=7.5, step=0.5, label="Guidance scale",
                                                        interactive=True)
                        num_inference_steps_for_vid = gr.Slider(10, 50, value=50, step=1, label="Number of inference steps",
                                                                interactive=True)
                        num_frames = gr.Slider(label='Number of frames', minimum=16, maximum=32, step=8, value=24,
                                            interactive=True,
                                            info='Note that the content of the video also changes when you change the number of frames.')
                    with gr.Accordion('Video Editing Advanced Options', open=False):
                        num_inference_steps_for_vid_edit = gr.Slider(10, 50, value=50, step=1, label="Number of inference steps",
                                                                interactive=True)
                        guidance_scale_for_vide_edit = gr.Slider(1, 100, value=50, step=10, label="The audio length in seconds",
                                                    interactive=True)
                    configs = [
                        seed, top_p, temperature,
                        guidance_scale_for_img_edit, num_inference_steps_for_img_edit,
                        guidance_scale_for_vid, num_inference_steps_for_vid, num_frames, 
                        num_inference_steps_for_vid_edit, guidance_scale_for_vide_edit
                    ]
                with gr.Tab("Operation"):
                    with gr.Row(scale=1):
                        submitBtn = gr.Button(value="Submit & Run", variant="primary")
                    with gr.Row(scale=1):
                        resubmitBtn = gr.Button("Rerun")
                    with gr.Row(scale=1):
                        emptyBtn = gr.Button("Clear History") 
        # input_image.upload(fn=clear_fn2, inputs=emptyBtn, outputs=[output_text, out_imagebox, input_image_state])
        # input_image.clear(fn=clear_fn2, inputs=emptyBtn, outputs=[output_text, out_imagebox, input_image_state])

            history = gr.State([])

        input_image.edit(
            fn=bbox_draw,
            inputs=[input_image, input_image_state],
            outputs=[out_imagebox, input_image_state],
            queue=False,
        )
        clearImageBtn.click(fn=clear_image_and_sketch_pad, inputs=[input_image_state], outputs=[input_image_state, input_image, out_imagebox])

        input_video.upload(fn=extract_frames, inputs=[input_video, input_video_state], outputs=[video_sketch_pad, input_video_state])
        video_sketch_pad.edit(fn=edit_video_frame, inputs=[video_sketch_pad, input_video_state], outputs=[video_sketch_pad, input_video_state])  # update the states
        nextFrameBtn.click(fn=select_next_frame, inputs=[input_video_state], outputs=[video_sketch_pad, input_video_state])
        clearFrameBtn.click(fn=clear_video_and_frame, inputs=[input_video_state], outputs=[input_video_state, video_sketch_pad, input_video])

        emptyBtn.click(fn=reset_state, inputs=[input_image_state, input_video_state], outputs=[input_image, out_imagebox, input_video, video_sketch_pad, history, chatbot])
        submitBtn.click(fn=predict, 
                        inputs=[tokenizer, model, image_processor, video_processor, conv, history, chatbot, user_input, 
                                input_image_state, input_image, out_imagebox,
                                input_video_state, input_video, video_sketch_pad,
                                configs], 
                        outputs=[chatbot, history]).then(clear_input, inputs=[input_image_state, input_video_state], outputs=[input_image_state, input_video_state, input_image, input_video, video_sketch_pad, out_imagebox],
                        show_progress=True
                        )
        resubmitBtn.click(fn=re_predict, 
                        inputs=[tokenizer, model, image_processor, video_processor, conv, history, chatbot, user_input, 
                                input_image_state, input_image, out_imagebox,
                                input_video_state, input_video, video_sketch_pad,
                                configs], 
                        outputs=[chatbot, history]).then(clear_input, inputs=[input_image_state, input_video_state], outputs=[input_image_state, input_video_state, input_image, input_video, video_sketch_pad, out_imagebox],
                        show_progress=True
                        )
    
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model_path", type=str, default="vitron")
    parser.add_argument("--model_base", type=str, default='vitron')
    parser.add_argument("--model_name", type=str, default='vitron')
    args = parser.parse_args()
    # LLAVA.set_args(args)
    tokenizer, model, image_processor, video_processor, conv = load_model(model_path=args.model_path, model_base=args.model_base, model_name=args.model_name)
    demo = build_demo()
    demo.queue().launch(server_name=args.host, server_port=args.port, share=True)

