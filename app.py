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
from app_utils import ImageBoxState, bbox_draw, open_image, mask_to_bbox
import imageio
import tempfile
from omegaconf import OmegaConf
import torch
from diffusers import I2VGenXLPipeline
from diffusers.utils import export_to_gif, export_to_video
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from urllib.request import urlopen
from PIL import Image


os.environ["BASE_HOME"] = os.path.dirname(__file__)
sys.path.append(os.path.dirname(__file__))

sys.path.append(os.path.join(os.environ['BASE_HOME'], 'modules/GLIGEN/demo'))
import modules.GLIGEN.demo.app as GLIGEN
import modules.GLIGEN.demo.gligen.task_grounded_generation as GLIGEN_generation

sys.path.append(os.path.join(os.environ['BASE_HOME'], 'modules/SEEM/demo_code'))
sys.path.append(os.path.join(os.environ['BASE_HOME'], 'modules/SEEM/demo_code/tasks'))
import modules.SEEM.demo_code.app as SEEM
import modules.SEEM.demo_code.utils.visualizer as visual
# import SEEM.demo_code.tasks.visualizer as visual

sys.path.append(os.path.join(os.environ['BASE_HOME'], 'modules/StableVideo'))
import modules.StableVideo.app as stablevideo

# sys.path.append(os.path.join(os.environ['BASE_HOME'], 'Vitron/vitron'))
from vitron.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, OBJS_TOKEN_INDEX, DEFAULT_VIDEO_TOKEN, DEFAULT_OBJS_TOKEN
from vitron.conversation import conv_templates, SeparatorStyle
from vitron.model.builder import load_pretrained_model
from vitron.utils import disable_torch_init
from vitron.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, tokenizer_image_region_token, preprocess_region, show_image_with_bboxes


def load_model(model_base, model_path, model_name='vitron-7b-lora-6'):
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

model_path = 'checkpoints/Vitron-lora/vitron-7b-lora-6'
model_base = 'checkpoints/Vitron-base'
model_name = 'vitron-7b-llava-lora-6'
tokenizer, model, image_processor, video_processor, conv = load_model(model_path=model_path, model_base=model_base, model_name=model_name)
print('load model successfully')


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
    # export_to_video(video, filename)
    writer = imageio.get_writer(filename, format='FFMPEG', fps=8)
    for frame in video:
        if isinstance(frame, Image.Image):
            frame = np.array(frame)
        writer.append_data(frame)
    writer.close()
    return filename


def image_generation(prompt="a black swan swimming in a pond surrounded by green plants"):
    """
    :param prompt: text
    :return: 
    """
    # sys.path.append(os.path.join(os.environ['BASE_HOME'], 'GLIGEN/demo'))
    # import GLIGEN.demo.gligen.task_grounded_generation as GLIGEN_generation
    cache_file = os.path.join('checkpoints/gligen/gligen-generation-text-box', 'diffusion_pytorch_model.bin')
    pretrained_ckpt_gligen = torch.load(cache_file, map_location='cpu')
    cache_config = os.path.join('checkpoints/gligen/demo_config_legacy', 'gligen-generation-text-box.pth')
    config = torch.load(cache_config, map_location='cpu')
    config = OmegaConf.create(config["_content"])
    config.update(
        {'folder': 'create_samples', 'official_ckpt': 'ckpts/sd-v1-4.ckpt', 'guidance_scale': 5, 'alpha_scale': 1})
    config.model['params']['is_inpaint'] = False
    config.model['params']['is_style'] = True
    loaded_model_list = GLIGEN_generation.load_ckpt(config, pretrained_ckpt_gligen)
    # phrase_list = []
    # placeholder_image = Image.open('images/teddy.jpg').convert("RGB")
    # image_list = [placeholder_image] * len(phrase_list)
    instruction = dict(prompt=prompt, save_folder_name='gen_res', batch_size=1,
                       phrases=['placeholder'], has_text_mask=1, has_image_mask=0,
                       images=[], alpha_type=[0.3, 0, 0.7], guidance_scale=7.5, fix_seed=True,
                       rand_seed=0, actual_mask=None, inpainting_boxes_nodrop=None, locations=[])
    sample_list, over_list = GLIGEN_generation.grounded_generation_box(loaded_model_list, instruction)
    image_path = save_image_to_local(sample_list[0])
    print(f'Generated image save into {image_path}')
    return image_path


def image_segmentation(image_path, track_text, sketch_pad=None):
    """
    Based on the input image, we segment the image and return the segmented image.
    Args:
        image (Image): The input image.
        track_text (str): The reference text.
        sketch_pad (Dict):
            ['image']: array
            ['mask']: array
    Returns:
        Image: The segmented image.
    """
    print('Calling SEEM_app.inference')
    # sys.path.append(os.path.join(os.environ['BASE_HOME'], 'SEEM/demo_code'))
    # sys.path.append(os.path.join(os.environ['BASE_HOME'], 'SEEM/demo_code/tasks'))
    # import SEEM.demo_code.app as SEEM
    # import SEEM.demo_code.utils.visualizer as visual
    # # import SEEM.demo_code.tasks.visualizer as visual

    img = open_image(image_path)
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
            task = ['Text']
        else:
            compose_img = {'image': open_image(sketch_pad['image']), 'mask': sketch_pad['image']}
            # print('image segmentation / sketch_pad', sketch_pad)  # sketch_pad['image']: array,  sketch_pad['mask']: array
            width, height = compose_img['image'].width, compose_img['image'].height
            task = ['Stroke']
        # task = ['Stroke']
        image, masks, labels = SEEM.inference(image=compose_img, task=task, reftxt=track_text)
        # print('image', image)
        # print('masks', masks)
        # print('labels', labels)
        mask_pred = masks[0].astype("uint8")
        mask_pred = cv2.resize(mask_pred, (width, height), interpolation=cv2.INTER_LANCZOS4)
        mask_pred = mask_pred.astype("uint8")
        print('mask_pred: ', mask_pred)
        # print('mask shape: ', len(mask_pred), len(mask_pred[0]))
        # print('height: ', height)
        # print('width: ', width)
        mask_demo = visual.GenericMask(mask_pred, height, width)
        # print('mask_demo: ', mask_demo)
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
    # sys.path.append(os.path.join(os.environ['BASE_HOME'], 'GLIGEN/demo'))
    # import GLIGEN.demo.app as GLIGEN
    image = open_image(image_path)
    width, height = image.size
    text = prompt
    texts = [x.strip() for x in text.split(';')]
    boxes = []
    masks = []
    if sketch_pad is None:
        # if there is no sketch_pad, i.e., no specification for image editing. Thus, first segmenting the image based on text， then inpainting the image
        for t in texts:
            _, t_mask, _ = image_segmentation(image_path=image_path, track_text=t)
            boxes.append(t_mask['boxes'])
            masks.append(t_mask['mask'])
        state = {'boxes': boxes}
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
        boxes = mask_to_bbox(sketch_pad['mask'])
        state = {'boxes': [boxes]}
        print('state: ', state)
        gen_images, state_list = GLIGEN.generate(task='Grounded Inpainting', language_instruction=prompt,
                                                 sketch_pad=sketch_pad,
                                                 grounding_texts=prompt, alpha_sample=1.0, guidance_scale=30,
                                                 batch_size=1, fix_seed=False, rand_seed=0, use_actual_mask=False,
                                                 append_grounding=False,
                                                 style_cond_image=None, state=state, inpainting_mask=None,
                                                 inpainting_image=image
                                                 )

    # gen_images[0].save(save_path)
    image_path = save_image_to_local(gen_images[0])
    print(f'Generated image save into {image_path}')
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
    # sys.path.append(os.path.join(os.environ['BASE_HOME'], 'Zeroscope'))
    pipe = DiffusionPipeline.from_pretrained("checkpoints/zeroscope", torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # prompt = prompt
    video_frames = pipe(prompt, num_inference_steps=num_inference_steps, 
                        guidance_scale=guidance_scale,
                        height=320, width=576, 
                        num_frames=num_frames).frames[0]
    # print('video_frames: ', video_frames)
    video_frames = [np.array(frame) for frame in video_frames]
    video_path = save_video_to_local(video_frames)
    print(f'Generated video save into {video_path}')
    # video_path = export_to_video(video_frames, f"{save_dir}/video.mp4")
    return video_path


def video_tracking(video_path="vasedeck.mp4", sketch_pad=None, track_prompt="", text_prompt=""):
    """
    Based on the input video, we track the video and return the tracked video.
    Args:
        video_path (str): The input video path.
        track_text (str): The track text.
        sketch_pad (dict): The sketchpad input with format {'image': Image, 'mask': Image}.
        track_prompt (str): The track prompt.
        text_prompt (str): The text prompt.  
            if no sketchpad, the text prompt is used to segment the image, obtaining the foreground images.
    Returns:
        str: The tracked video path.
    """
    # sys.path.append(os.path.join(os.environ['BASE_HOME'], 'SEEM/demo_code'))
    # import SEEM.demo_code.app as SEEM
    if sketch_pad is None:
        i_video_path = video_path.split('/')[-2]
        img, o = video_editing(video_path=i_video_path, fore_prompt=text_prompt, back_prompt="")
        image_path = save_image_to_local(img)
        img = Image.open(image_path)
        compose_img = {'image': img, 'mask': img}
    else:
        # compose_img = sketch_pad
        compose_img = {'image': open_image(sketch_pad['image']), 'mask': sketch_pad['image']}

    _, output_video_name = SEEM.inference("examples/placeholder.png", task=['Video'],
                                          video_pth=video_path, refimg=compose_img, reftxt=track_prompt)
    return output_video_name


def video_editing(video_path="The_test_new_video.mp4", fore_prompt="turn the orange into bread",
               back_prompt="change the background into the blue"):
    """

    :param video_path (str): directory of the video to be modified. The file structure shoud be like this
        ./data/xxx
            -- checkpoint
            -- config.json
            -- xxx.mp4
            -- texture.orig1.png
            -- texture.orig2.png
            -- xxx
                -- 00000.jpg
                -- 00001.jpg

    :param fore_prompt (str): prompt for modifying foreground, such as "turn the orange into bread"
    :param back_prompt (str): prompt for modifying background, such as "change the background into the blue"
    """
    # sys.path.append(os.path.join(os.environ['BASE_HOME'], 'StableVideo'))
    # import StableVideo.app as stablevideo

    st = stablevideo.StableVideo(base_cfg="checkpoints/stablevideo/cldm_v15.yaml",
                                 canny_model_cfg="checkpoints/stablevideo/control_sd15_canny.pth",
                                 depth_model_cfg="checkpoints/stablevideo/control_sd15_depth.pth",
                                 save_memory=False)

    st.load_canny_model(base_cfg='checkpoints/stablevideo/cldm_v15.yaml',
                        canny_model_cfg='checkpoints/stablevideo/control_sd15_canny.pth')
    st.load_depth_model(base_cfg='checkpoints/stablevideo/cldm_v15.yaml',
                        depth_model_cfg='checkpoints/stablevideo/control_sd15_depth.pth', )
    video_save_name, f_atlas_origin, b_altas_origin = st.load_video(video_path=video_path, video_name=os.path.basename(video_path))
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
    # sys.path.append(os.path.join(os.environ['BASE_HOME'], 'i2vgen-xl'))
    pipe = I2VGenXLPipeline.from_pretrained("checkpoints/i2vgen-xl", torch_dtype=torch.float16, variant="fp16")
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
    video_path = save_video_to_local(frames)
    print(f'Generated video save into {video_path}')
    return video_path


def find_module_content(data):
    pattern = r'<module>(.*?)</module>'
    match = re.search(pattern, data)
    if match:
        return match.group(1)
    else:
        return None


def find_instruction_content(data):
    pattern = r'<instruction>(.*?)</instruction>'
    match = re.findall(pattern, data)
    
    if match:
        res = []
        for _res in match:
            res.append(_res.split(':')[-1].strip())
        return res
    else:
        return None


def find_region_instrction_content(data):
    pattern = r'<region>(.*?)</region>'
    match = re.search(pattern, data)
    if match:
        return match.group(1)
    else:
        return None


def remove_special_tags(text):
    """
    remove the content between the tags and also the tags: <module></module> <instruction></instruction> <region></region> <SP></SP>
    """
    pattern = r'<[^>]+>(.*?)<[^>]+>'  # match all the tags
    return re.sub(pattern, '', text)


def parse_model_output(model_output):
    """
    Based on the model output, we parse the model output and return the parsed instructions.
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
        query (list): The input query. query[0]-> text, query[1]-> image, query[2]-> video
        video_processor (VideoProcessor): The video processor.  
        image_processor (ImageProcessor): The image processor.
    """
    res_utterance = ''
    video_tensor = None
    image_tensor = None
    region = None
    if query[1] is not None and query[2] is not None:
        # input includes video and image
        res_utterance = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + ' ' + DEFAULT_IMAGE_TOKEN + '\n' + query[0]
        image_tensor = image_processor.preprocess(query[1], return_tensors='pt')['pixel_values'][0]
        video_tensor = video_processor(query[2], return_tensors='pt')['pixel_values']
        region = query[3]
    elif query[1] is not None and query[2] is None:
        # input includes image but no video
        res_utterance = DEFAULT_IMAGE_TOKEN + '\n' + query[0]
        image_tensor = image_processor.preprocess(query[1], return_tensors='pt')['pixel_values'][0]
        region = query[3]
    elif query[1] is None and query[2] is not None:
        # input includes no image but video
        res_utterance = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + query[0]
        video_tensor = video_processor(query[2], return_tensors='pt')['pixel_values'][0]
        region = query[3]
    else:
        # input includes no video but image
        res_utterance = query[0]
    return res_utterance, video_tensor, image_tensor, region


def re_predict(user_input, input_image_state, input_image, out_imagebox,
             input_video_state, input_video, video_sketch_pad, history, chatbox, 
            configs):
    q, a = history.pop()
    chatbox.pop()
    return predict(q, input_image_state, input_image, out_imagebox,
             input_video_state, input_video, video_sketch_pad, history, chatbox, 
            configs)


def predict(user_input, input_image_state, input_image, out_imagebox,
            input_video_state, input_video, video_sketch_pad, history, chatbox, 
            *args):
    """
    Based on the user input and history, we generate the response and update the history.
    Args:
        tokenizer (Tokenizer): 
            The tokenizer for process user input instructions.
        model (Model): The model.
        image_processor (ImageProcessor): 
            The image processor for process image.
        video_processor (VideoProcessor): 
            The video processor for process video.
        conv(Conversation): Conversation class.
        history (list): 
            The history. [[(q1, v1, i1, r1), (a1, v1, i1, r1)], [(q2, v2, i2, r2), (a2, v2, i2, r2)]
        chatbox (list): The chatbox.
        input_image_state (dict): {'ibs': ImageBoxState}.
            Saving the image state including the image, box and mask.
        input_image (Numpy.ndarray): 
            The input image.
        out_imagebox (dict): 
            The output image box. {'image': Numpy.ndarray, 'mask': Numpy.ndarray}.
        input_video_state (dict): {'ibs': ImageBoxState}.
            Saving the video state including the video frame list, current frame, box and mask.
        input_video (str): 
            The file path of input video.
        video_sketch_pad (dict): 
            The video sketch pad of each frame. {'image': Numpy.ndarray, 'mask': Numpy.ndarray}.
        configs (dict): The configurations.

    """
    video_tensors = []
    image_tensors = []
    input_region = []
    config = create_cfg(*args)
    if history is not None:
        print('history: ', history)
        default_input_region = [0, 0, 224, 224]
        for idx, _his in enumerate(history):
            print(f'idx: {idx},  history[idx]: {_his}')
            q, a = _his
            q_utterance, q_video_tensor, q_image_tensor, q_region = get_utterence(q, video_processor, image_processor)
            conv.append_message(conv.roles[0], q_utterance)
            a_utterance, a_video_tensor, a_image_tensor, a_region = get_utterence(a, video_processor, image_processor)
            conv.append_message(conv.roles[1], a_utterance)
            if q_video_tensor is not None:
                video_tensors.append(q_video_tensor)
                input_region.append(q_region)
            if q_image_tensor is not None:
                image_tensors.append(q_image_tensor)
                input_region.append(q_region)
            if a_video_tensor is not None:
                video_tensors.append(a_video_tensor)
                input_region.append(q_region)
            if a_image_tensor is not None:
                image_tensors.append(a_image_tensor)
                input_region.append(a_region)
        
    inp = ''
    _user_input = user_input
    query_img_path = ''
    if input_video is not None:
        inp = inp + ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames)
        video_tensors.append(video_processor(input_video, return_tensors='pt')['pixel_values'][0])  # 'input_video' should be a file_path
        _user_input += f'<br><video controls playsinline width="500" style="display: inline-block;"  src="./file={input_video}"></video>'
        input_region.append(default_input_region)
    if input_image_state['ibs'].image is not None:
        inp = inp + ' ' + DEFAULT_IMAGE_TOKEN
        image_tensors.append(image_processor.preprocess(input_image_state['ibs'].image, return_tensors='pt')['pixel_values'][0])
        if len(input_image_state['ibs'].boxes) > 0:
            bbox = input_image_state['ibs'].boxes[-1]
            input_region.append(bbox)
            ori_im_size = [input_image_state['ibs'].width, input_image_state['ibs'].height]
            input_region = [preprocess_region(_bbox, ori_im_size, [224, 224]) for _bbox in input_region]
            inp = inp + '\n' + DEFAULT_OBJS_TOKEN + ' '
        else:
            input_region.append(default_input_region)
        query_img_path = save_image_to_local(input_image_state['ibs'].image)
        _user_input += f'<br><img src="./file={query_img_path}" style="display: inline-block;width: 250px;max-height: 400px;">'
    inp = inp + '\n' + user_input if inp.endswith('>') else inp + user_input
    print('inp: ', inp)
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    conv.clear_message()

    print('prompt: ', prompt)

    # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    input_ids = tokenizer_image_region_token(prompt, tokenizer, OBJS_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    if len(image_tensors) == 0 and len(video_tensors) == 0:
        # no image or video input 
        tensor = [torch.zeros(3, image_processor.crop_size['height'], image_processor.crop_size['width']).to(model.device, dtype=torch.float16)]
        input_region = [default_input_region]
    else:
        tensor =  [_tensor.to(model.device, dtype=torch.float16) for _tensor in video_tensors+image_tensors]
        assert len(input_region) == len(tensor)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor,
            regions = input_region,
            do_sample=True,
            temperature=config['temperature'],
            top_p = config['top_p'],
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    print('model outputs: ', outputs)
    outputs = 'Absolutely! In the video, a muscular man is performing a fire show with two rotating fireballs on a chain. I will edit the video to make it look like the man has wings and is performing the fire show in the sky. <module>F</module> <instruction> foreground: add wings to the man and position him in the sky </instruction> <instruction> background: keep </instruction> <SP>None</SP>'  # video editing
    output, module, instruction, region = parse_model_output(outputs)
    print('parsed output: ', output)
    print('module: ', module)
    print('instruction: ', instruction)
    print('region: ', region)
    # _user_input = None
    if input_image_state['ibs'].image is not None:
        query_img_path = save_image_to_local(input_image_state['ibs'].image)
    else:
        query_img_path = None

    if module and module in tasks:
        if module == 'A':
            if instruction is not None and len(instruction) > 0:
                ans_image_path = image_generation(prompt=instruction[0])
                _response = output + f'<br><img src="./file={ans_image_path}" style="display: inline-block;width: 250px;max-height: 400px;">'
            else:
                _response = output
                ans_image_path = None 
            chatbox.append((_user_input, _response))
            history.append(((user_input, query_img_path, input_video, input_region[-1]), (output, ans_image_path, None, default_input_region)))
        elif module == 'B':
            if instruction is not None and len(instruction) > 0:
                image_seg, pad, label = image_segmentation(image_path=query_img_path, track_text=instruction[0],
                                                            sketch_pad=input_image)
                ans_image_path = save_image_to_local(image_seg)
                _response = output + f'<br><img src="./file={ans_image_path}" style="display: inline-block;width: 250px;max-height: 400px;">'
            else:
                _response = output
                ans_image_path = None   
            print(f'image file save into {ans_image_path}')
            chatbox.append((_user_input, _response))    
            history.append(((user_input, query_img_path, input_video, input_region[-1]), (output, ans_image_path, None, default_input_region)))
        elif module == 'C':
            if instruction is not None and len(instruction) > 0:
                ans_image_path, _ = image_editing(image_path=query_img_path, sketch_pad=input_image, prompt=instruction[0])
                _response = output + f'<br><img src="./file={ans_image_path}" style="display: inline-block;width: 250px;max-height: 400px;">'
            else:
                _response = output
                ans_image_path = None
            chatbox.append((_user_input, _response))
            history.append(((user_input, query_img_path, input_video, input_region[-1]), (output, ans_image_path, None, default_input_region)))
        elif module == 'D':
            if instruction is not None and len(instruction) > 0:
                ans_video_path = video_generation(prompt=instruction[0], num_inference_steps=config['num_inference_steps_for_vid'], num_frames=config['num_frames'],  guidance_scale=config['guidance_scale_for_vid'])
                _response = output + f'<br><video controls playsinline width="500" style="display: inline-block;"  src="./file={ans_video_path}"></video>'
            else:
                _response = output
                ans_image_path = None
            chatbox.append((_user_input, _response))
            history.append(((user_input, query_img_path, input_video, input_region[-1]), (output, None, ans_video_path, default_input_region)))
        elif module == 'E':
            query_img_path = save_image_to_local(input_video_state['ibs'].image)
            if instruction is not None and len(instruction) > 0:
                # target_video_path = os.path.join('./temp', os.path.basename(input_video))
                # cp_cmd = 'cp {} {}'.format(input_video, target_video_path)
                # os.system(cp_cmd)
                ans_video_path = video_tracking(video_path=input_video, sketch_pad=video_sketch_pad, track_prompt=instruction[0])
                _response = output + f'<br><video controls playsinline width="500" style="display: inline-block;"  src="./file={ans_video_path}"></video>'
            else:
                _response = output
                ans_video_path = None
            chatbox.append((_user_input, _response))
            history.append(((user_input, query_img_path, input_video, input_region[-1]), (output, None, ans_video_path, default_input_region)))
        elif module == 'F':
            if instruction is not None and len(instruction) >= 2:
                _, ans_video_path = video_editing(video_path=input_video, fore_prompt=instruction[0], back_prompt=instruction[1])
                _response = output + f'<br><video controls playsinline width="500" style="display: inline-block;"  src="./file={ans_video_path}"></video>'
            else:
                _response = output
                ans_video_path = None
            chatbox.append((_user_input, _response))
            history.append(((user_input, query_img_path, input_video, input_region[-1]), (output, None, ans_video_path, default_input_region)))
        elif module == 'G':
            if instruction is not None and len(instruction) > 0:
                ans_video_path = image_to_video(image_path=query_img_path, text_prompt=instruction[0])
                _response = output + f'<br><video controls playsinline width="500" style="display: inline-block;"  src="./file={ans_video_path}"></video>'
            else:
                _response = output
                ans_video_path = None
            chatbox.append((_user_input, _response))
            history.append(((user_input, query_img_path, input_video, input_region[-1]), (output, None, ans_video_path, default_input_region)))
        else:
            raise NotImplementedError(f'The module {module} is not implemented.')
    else:
        chatbox.append((_user_input, output))
        history.append(((user_input, query_img_path, input_video, input_region[-1]), (output, None, None, default_input_region)))

    return chatbox, history, None, None, None, None


def new_state():
    return {"ibs": ImageBoxState()}


def clear_fn2(value):
    return default_chatbox, None, new_state()


def upload_image(sketch_pad: dict, state: dict):
    image = sketch_pad['image']  
    # print('sketch_pad', sketch_pad)  # {'image': array unit8, 'mask': array unit8}
    image = open_image(image)
    ibs = state["ibs"]
    ibs.update_image(image)
    return image, state


def reset_state(input_image_state, input_video_state):
    ibs = input_image_state["ibs"]
    ibs.reset_state()
    
    ibs = input_video_state["ibs"]
    ibs.reset_state()

    return None, None, None, None, None, [], [], []


def create_cfg(seed, top_p, temperature,
            guidance_scale_for_img_edit, num_inference_steps_for_img_edit,
            guidance_scale_for_vid, num_inference_steps_for_vid, num_frames, 
            num_inference_steps_for_vid_edit, guidance_scale_for_vide_edit):
    cfg_dict = {
        "seed": seed,
        "top_p": top_p,
        "temperature": temperature,
        "guidance_scale_for_img_edit":guidance_scale_for_img_edit, 
        "num_inference_steps_for_img_edit": num_inference_steps_for_img_edit,
        "guidance_scale_for_vid": guidance_scale_for_vid,
        "num_inference_steps_for_vid": num_inference_steps_for_vid,
        "num_frames": num_frames,
        "num_inference_steps_for_vid_edit": num_inference_steps_for_vid_edit,
        "guidance_scale_for_vide_edit": guidance_scale_for_vide_edit
    }
    return cfg_dict

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
    return image_state, video_state, None, None, None, None, None


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
                    chatbot = gr.Chatbot(label='Vitron Chatbot', height=500, elem_id='chatbox', avatar_images=((os.path.join(os.path.dirname(__file__), 'user.png')), (os.path.join(os.path.dirname(__file__), "vitron.png"))))

                with gr.Row():
                    user_input = gr.Textbox(label='User Input', placeholder='Enter your text here', elem_id='user_input', lines=3)  
                
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Tab('🏞️ Image'):
                            with gr.Row():
                                # input_image = gr.Image(label='Input Image', type='numpy',
                                #                         shape=(512, 512), 
                                #                         # height=200, width=200, 
                                #                         elem_id='img2img_image', 
                                #                         interactive=True, tool='sketch', 
                                #                         brush_radius=20.0, visible=True)
                                input_image = ImageMask(label="Input Image", type="numpy",
                                                        shape=(512, 512), 
                                                        # height=200, width=200, 
                                                        elem_id='img2img_image',
                                                        # tool='sketch', 
                                                        brush_radius=20.0, visible=True)
                            with gr.Row():
                                clearImageBtn = gr.Button("Clear Image and Sketch Pad", elem_id='clear_image')
                            with gr.Row():
                                out_imagebox = gr.Image(label='Parsed Sketch Pad"', type='numpy',
                                                shape=(512, 512),  
                                                # height=200, width=200, 
                                                elem_id='out_imagebox')
                                input_image_state = gr.State(new_state())
                            
                    with gr.Column(scale=3):
                        with gr.Tab('🎬 Video'):
                            with gr.Row():
                                input_video = gr.Video(label='Input Video', format='mp4', visible=True)  #.style(height=200) # , value=None, interactive=True
                            with gr.Row():
                                with gr.Column(scale=0.3):
                                    nextFrameBtn = gr.Button("Next Frame", elem_id='next_frmame', variant="primary")
                                    clearFrameBtn = gr.Button("Clear Video & Frame", elem_id='clear_frmame')
                                # with gr.Column(scale=0.3):
                                    
                            with gr.Row():
                                # video_sketch_pad = gr.Image(label='Video Frame', type='numpy', 
                                #                     shape=(512, 512),
                                #                     # height=200, width=200, 
                                #                     elem_id='video_sketch_pad', 
                                #                     interactive=True, tool='sketch', 
                                #                     brush_radius=20.0, visible=True)
                                video_sketch_pad = ImageMask(label='Video Frame', type='numpy', 
                                                    shape=(512, 512),
                                                    # height=200, width=200, 
                                                    elem_id='video_sketch_pad', 
                                                    # interactive=True, tool='sketch', 
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
                with gr.Tab("🎯 Operation"):
                    with gr.Row(scale=1):
                        submitBtn = gr.Button(value="Submit & Run", variant="primary")
                    with gr.Row(scale=1):
                        resubmitBtn = gr.Button("Rerun")
                    with gr.Row(scale=1):
                        emptyBtn = gr.Button("Clear History") 
                
        # input_image.upload(fn=clear_fn2, inputs=emptyBtn, outputs=[output_text, out_imagebox, input_image_state])
        # input_image.clear(fn=clear_fn2, inputs=emptyBtn, outputs=[output_text, out_imagebox, input_image_state])

            history = gr.State([])

        input_image.upload(fn=upload_image, inputs=[input_image, input_image_state], outputs=[out_imagebox, input_image_state])
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

        emptyBtn.click(fn=reset_state, inputs=[input_image_state, input_video_state], outputs=[user_input, input_image, out_imagebox, input_video, video_sketch_pad, history, chatbot])
        submitBtn.click(fn=predict, 
                        inputs=[user_input, input_image_state, input_image, out_imagebox,
                                input_video_state, input_video, video_sketch_pad, history, chatbot, *configs
                                ], 
                        outputs=[chatbot, history], show_progress=True).then(clear_input, inputs=[input_image_state, input_video_state], outputs=[input_image_state, input_video_state, input_image, input_video, video_sketch_pad, out_imagebox, user_input],
                        show_progress=True
                        )
        resubmitBtn.click(fn=re_predict, 
                        inputs=[user_input, input_image_state, input_image, out_imagebox,
                                input_video_state, input_video, video_sketch_pad, history, chatbot, 
                                ], 
                        outputs=[chatbot, history], show_progress=True).then(clear_input, inputs=[input_image_state, input_video_state], outputs=[input_image_state, input_video_state, input_image, input_video, video_sketch_pad, out_imagebox, user_input],
                        show_progress=True
                        )
    
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    demo = build_demo()
    demo.queue().launch(server_name=args.host, server_port=args.port, share=True)



