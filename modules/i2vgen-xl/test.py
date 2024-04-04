import torch
from diffusers import I2VGenXLPipeline
from diffusers.utils import export_to_gif, export_to_video
from urllib.request import urlopen
from PIL import Image


pipeline = I2VGenXLPipeline.from_pretrained("/mnt/haofei/VideoGPT/LLaVA-Interactive-Demo/i2vgen-xl/checkpoints", torch_dtype=torch.float16, variant="fp16")
pipeline.enable_model_cpu_offload()

# image_url = "https://github.com/ali-vilab/i2vgen-xl/blob/main/data/test_images/img_0009.png?raw=true"
# image = load_image(image_url).convert("RGB")
image = Image.open('./data/test_images/street.png').convert("RGB")

prompt = "a car on the road, a white dog is beside the car, the dog run past the car"
negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
generator = torch.manual_seed(8800)

frames = pipeline(
    prompt=prompt,
    image=image,
    num_inference_steps=50,
    negative_prompt=negative_prompt,
    guidance_scale=9.0,
    generator=generator
).frames[0]
print(frames)
for idx, img in enumerate(frames):
    img.save(f'street/00{idx}.jpg')
video_path = export_to_gif(frames, "street.gif")
video_path = export_to_video(frames, "street_1.mp4")
print(video_path)
