import torch
import os
from PIL import Image
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, OBJS_TOKEN_INDEX
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, tokenizer_image_region_token, preprocess_region, show_image_with_bboxes

def inference_image():
    disable_torch_init()
    # image = 'videollava/serve/examples/extreme_ironing.jpg'
    image = '/mnt/haofei/VideoGPT/LLaVA-Interactive-Demo/data/coco2017/train2017/000000020150.jpg'
    # image = 'videollava/serve/examples/desert.jpg'
    # inp = 'What is unusual about this image? '
    # inp = 'Can you give me a description of the region <objs> in image?'
    # inp = 'Can you modified the marked car into a train in image?'
    inp = 'Could you help me pinpoint the zebra that is farthest from the viewer in this image?'
    model_base = '/mnt/haofei/VideoGPT/LLaVA-Interactive-Demo/LanguageBind/Video-LLaVA-7B'
    # model_path = '/mnt/haofei/VideoGPT/LLaVA-Interactive-Demo/LanguageBind/Video-LLaVA-7B'
    cache_dir = 'cache_dir'
    device = 'cuda'
    load_4bit, load_8bit = False, False
    model_path = '/mnt/haofei/VideoGPT/LLaVA-Interactive-Demo/VideoLLaVA/checkpoints/videollava-7b-lora-7'
    # model_name = get_model_name_from_path(model_path)
    # _weights = torch.load(os.path.join(model_base, 'adapter_model.bin'), map_location='cpu')
    # # _weights = torch.load(os.path.join(model_base, 'adapter_model.bin'), map_location='cpu')
    # _weights = torch.load(os.path.join(model_base, 'non_lora_trainables.bin'), map_location='cpu')
    # print('xxxx')
    tokenizer, model, processor, _ = load_pretrained_model(model_path, model_base, 'video-llava-7b-lora', load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    image_processor = processor['image']
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
    if type(image_tensor) is list:
        tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        tensor = image_tensor.to(model.device, dtype=torch.float16)

    # preprocess region
    ori_im_size = [Image.open(image).convert('RGB').width, Image.open(image).convert('RGB').height]
    print('ori_im_size: ', ori_im_size)  # [570, 380]
    bbox = [0,100,300,200]
    region = [preprocess_region(bbox, ori_im_size, [224, 224])]  
    print('image_tensor: ', image_tensor)
    show_image_with_bboxes(image_path=image, bboxes=[bbox], save_path=os.path.join(os.path.dirname(image), 'ann_1.jpg'))
    show_image_with_bboxes(image_path=image_tensor[0], bboxes=region, save_path=os.path.join(os.path.dirname(image), 'ann_2.jpg'))
    print(f"{roles[1]}: {inp}")
    print('model device', model.device)
    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print('prompt: ', prompt)
    # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    input_ids = tokenizer_image_region_token(prompt, tokenizer, OBJS_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    print('input_ids: ', input_ids)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor,
            regions = region,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    print(outputs)


def inference_video():
    disable_torch_init()
    video = 'videollava/serve/examples/sample_demo_1.mp4'
    inp = 'Why is this video funny?'
    # model_path = 'LanguageBind/Video-LLaVA-7B'
    model_base = '/mnt/haofei/VideoGPT/LLaVA-Interactive-Demo/LanguageBind/Video-LLaVA-7B'
    cache_dir = 'cache_dir'
    device = 'cuda'
    load_4bit, load_8bit = False, False
    # model_name = get_model_name_from_path(model_path)
    model_path = '/mnt/haofei/VideoGPT/LLaVA-Interactive-Demo/VideoLLaVA/checkpoints/videollava-7b-lora-2'
    tokenizer, model, processor, _ = load_pretrained_model(model_path, model_base, 'video-llava-7b-lora', load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    video_processor = processor['video']
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    video_tensor = video_processor(video, return_tensors='pt')['pixel_values']
    if type(video_tensor) is list:
        tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to(model.device, dtype=torch.float16)

    print(f"{roles[1]}: {inp}")
    inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    bbox = [0,100,300,200]
    region = [preprocess_region(bbox, (480, 600), [224, 224])]
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor,
            regions = region,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    print(outputs)

if __name__ == '__main__':
    inference_image()
    # inference_video()