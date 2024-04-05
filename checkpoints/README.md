#  Checkpoints Preparation
Here, we provide the instructions and scripts for setting up the checkpoints employed in the Vitron.


## The List of Checkpoints

| Model  | Function  | Saving Path | Downloading Link |
| :------- |:---------------| :-----|  :-----:|
| GLIGEN   | image generation & editing | [.checkpoints/gligen](.checkpints/gligen) | [Link](https://github.com/gligen/GLIGEN) |
| i2vgen-xl     | image-to-video generation        |   [.checkpoints/i2vgen-xl](.checkpints/i2vgen-xl) | [Link](https://huggingface.co/ali-vilab) |
| LanguageBind | image & video encoder        |    [.checkpoints/LanguageBind](.checkpints/LanguageBind) | [Link](https://github.com/PKU-YuanGroup/LanguageBind) |
| OpenCLIP | image & text encoder        |    [.checkpoints/openai](.checkpints/openai)|[Link](https://huggingface.co/openai) |
| SEEM | image & video segmentation        |    [.checkpoints/seem](.checkpints/seem) | [Link]() |
| StableVideo | video editing        |    [.checkpoints/stablevideo](.checkpints/stablevideo) | [Link](https://github.com/rese1f/StableVideo) |
| Vitron-base |   reasoning      |    [.checkpoints/Vitron-base](.checkpints/Vitron-base)| [Link]() |
| Vitron-lora | reasoning       |    [.checkpoints/Vitron-lora](.checkpints/Vitron-lora)| [Link]() |
| ZeroScope | video generation       |    [.checkpoints/zeroscope](.checkpints/zeroscope) | [Link]() |




## The File Structure

```
checkpoints
├── gligen
│   ├── demo_config_legacy
│   │   ├── gligen-generation-text-box.pth
│   │   ├── gligen-generation-text-image-box.pth
│   │   └── gligen-inpainting-text-box.pth
│   ├── gligen-generation-text-box
│   │   └── diffusion_pytorch_model.bin
│   ├── gligen-generation-text-image-box
│   │   └── diffusion_pytorch_model.bin
│   └── gligen-inpainting-text-box
│       └── diffusion_pytorch_model.bin
├── i2vgen-xl
│   ├── feature_extractor
│   │   └── preprocessor_config.json
│   ├── image_encoder
│   │   ├── config.json
│   │   ├── model.fp16.safetensors
│   │   └── model.safetensors
│   ├── model_index.json
│   ├── models
│   │   ├── i2vgen_xl_00854500.pth
│   │   ├── open_clip_pytorch_model.bin
│   │   ├── stable_diffusion_image_key_temporal_attention_x1.json
│   │   └── v2-1_512-ema-pruned.ckpt
│   ├── scheduler
│   │   └── scheduler_config.json
│   ├── text_encoder
│   │   ├── config.json
│   │   ├── model.fp16.safetensors
│   │   └── model.safetensors
│   ├── tokenizer
│   │   ├── merges.txt
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   ├── unet
│   │   ├── config.json
│   │   ├── diffusion_pytorch_model.fp16.safetensors
│   │   └── diffusion_pytorch_model.safetensors
│   └── vae
│       ├── config.json
│       ├── diffusion_pytorch_model.fp16.safetensors
│       └── diffusion_pytorch_model.safetensors
├── LanguageBind
│   ├── LanguageBind_Image
│   │   ├──  ...
│   ├── LanguageBind_Video
│   │   ├──  ...
│   └── LanguageBind_Video_merge
│   │   ├──  ...
├── openai
│   ├── clip-vit-base-patch32
│   │   ├──  ...
│   └── clip-vit-large-patch14
│   │   ├──  ...
├── seem
│   └── seem_focall_v1.pt
├── stablevideo
│   ├── cldm_v15.yaml
│   ├── control_sd15_canny.pth
│   ├── control_sd15_depth.pth
│   ├── download.py
│   ├── dpt_hybrid-midas-501f0c75.pt
│   └── flan-t5-xl
│   │   ├──  ...
├── Vitron-base
│   ├── config.json
│   ├── generation_config.json
│   ├── pytorch_model-00001-of-00002.bin
│   ├── pytorch_model-00002-of-00002.bin
│   ├── pytorch_model.bin.index.json
│   ├── README.md
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.model
├── Vitron-lora
│   ├── vitron-7b-lora-4
│   │   ├── adapter_config.json
│   │   ├── adapter_model.bin
│   │   ├── config.json
│   │   ├── non_lora_trainables.bin
│   │   └── trainer_state.json
└── zeroscope
    ├── model_index.json
    ├── scheduler
    │   └── scheduler_config.json
    ├── text_encoder
    │   ├── config.json
    │   └── pytorch_model.bin
    ├── tokenizer
    │   ├── merges.txt
    │   ├── special_tokens_map.json
    │   ├── tokenizer_config.json
    │   └── vocab.json
    ├── unet
    │   ├── config.json
    │   └── diffusion_pytorch_model.bin
    ├── vae
    │   ├── config.json
    │   └── diffusion_pytorch_model.bin
    └── zs2_576w
        ├── open_clip_pytorch_model.bin
        └── text2video_pytorch_model.pth
```


## Downloading Checkpoints

To obtain the model checkpoints, you have two options: 
- First, you can manually download them using the provided links and place them in their respective directories as outlined above. 
- Alternatively, for a more automated approach, you can execute the scripts below:

```
cd checkpoints
bash download.sh
```