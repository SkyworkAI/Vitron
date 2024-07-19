


# Output-side Invocation-oriented Instruction Tuning



## Introduction

Instruction tuning for invocation aims to equip the system with the precise capability to execute commands, allowing the LLM to generate appropriate and correct invocation text.
Different terminal vision tasks might require distinct invocation commands. To unify this, we try to standardize the LLM’s response output into a structured text format, which includes: 
1. `User response output`, which directly replies to the user’s input. 
2. `Module name`, indicating which function or task is to be executed. 
3. `Invocation command`, a meta-instruction for triggering the
task module. 
4. `Region (optional)`, specifying a fine-grained vision feature needed for certain tasks, such as in video tracking or vision editing, where backend modules require this information.

For example:

```
[
    {
        "idx": 0,
        "source_data": {
            "image_name": "COCO_train2014_000000020150.jpg",
            "anno": "zebra farthest from you"
        },
        "image": "data/coco2017/train2017/000000020150.jpg",
        "bbox": [
            0,
            0,
            500,
            332
        ],
        "dataset_name": "coco2017",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nCould you help me pinpoint the zebra that is farthest from the viewer in this image?"
            },
            {
                "from": "gpt",
                "value": "Sure, I'll focus on the zebra that is the farthest from us. <module>B</module> <instruction>segmentation: zebra farthest from you</instruction>"
            }
        ]
    },
    ...
]
```

The summay of the constructed dataset are shown as follows:

<p align="center" width="100%">
<a target="_blank"><img src="../assets/image.png" alt="dataset" style="width: 90%; min-width: 200px; display: block; margin: auto;"></a>
</p>


## Preparation

To prepare the dataset, first you need to download the corresponding source dataset, including:
- **COCO2017** $\Longrightarrow$ `data/coco2017/train2017`
- **WebVid** $\Longrightarrow$  `data/webvid/dataset/`
- **Visual Genome** $\Longrightarrow$  `data/vg/`
- **MagicBruch** $\Longrightarrow$  `data/magicbrush/train/images`
- **CC3M** $\Longrightarrow$ `data/cc3m/cc3m/`

Then, put the image/video file in the corresponding directory.




