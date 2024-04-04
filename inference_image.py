import torch
import os
from PIL import Image
from vitron.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, OBJS_TOKEN_INDEX
from vitron.conversation import conv_templates, SeparatorStyle
from vitron.model.builder import load_pretrained_model
from vitron.utils import disable_torch_init
from vitron.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, tokenizer_image_region_token, preprocess_region, show_image_with_bboxes

def inference_image():
    disable_torch_init()
    image = 'examples/extreme_ironing.jpg'
    inp = 'Could you help me transform the image into a video?'
    model_base = 'checkpoints/Vitron-base'
    cache_dir = 'cache_dir'
    device = 'cuda'
    load_4bit, load_8bit = False, False
    model_path = 'checkpoints/Vitron-lora/'
    tokenizer, model, processor, _ = load_pretrained_model(model_path, model_base, 'vitron-llava-7b-lora-4', load_8bit, load_4bit, device=device, cache_dir=cache_dir)
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
    show_image_with_bboxes(image_path=image, bboxes=[bbox], save_path=os.path.join('./', 'ann_1.jpg'))
    show_image_with_bboxes(image_path=image_tensor[0], bboxes=region, save_path=os.path.join('./', 'ann_2.jpg'))
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
            # regions = region,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    print(outputs)


def inference_video():
    disable_torch_init()
    video = 'examples/sample_demo_1.mp4'
    inp = 'Why is this video funny?'
    # model_path = 'LanguageBind/Video-LLaVA-7B'
    model_base = 'checkpoints/Vitron-base'
    cache_dir = 'cache_dir'
    device = 'cuda'
    load_4bit, load_8bit = False, False
    model_path = 'checkpoints/Vitron-lora/'
    tokenizer, model, processor, _ = load_pretrained_model(model_path, model_base, 'vitron-llava-7b-lora-4', load_8bit, load_4bit, device=device, cache_dir=cache_dir)
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



# def test_modality_lengths(data_path):
#     import json
#     list_data_dict = json.load(open(data_path, "r"))
#     length_list = []
#     for sample in list_data_dict:
#         cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
#         cur_len = cur_len if 'image' in sample else -cur_len
#         length_list.append(cur_len)
#     return length_list

if __name__ == '__main__':
    inference_image()
    # inference_video()
    # test_modality_lengths('/mnt/haofei/VideoGPT/LLaVA-Interactive-Demo/data/checkpoint/preprocess_self_constructed_3.json')