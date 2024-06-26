echo "prepared checkpoints for GLIGEN"
mkdir gligen
cd gligen
git clone https://huggingface.co/gligen/demo_ckpts_legacy
git clone https://huggingface.co/gligen/gligen-generation-text-box
git clone https://huggingface.co/gligen/gligen-generation-text-image-box
git clone https://huggingface.co/gligen/gligen-inpainting-text-box


cd ..
echo "prepared checkpoints for i2vgen-xl"
git clone https://huggingface.co/ali-vilab/i2vgen-xl


cd ..
echo "prepared checkpoints for LanguageBind"
mkdir LanguageBind
git clone https://huggingface.co/LanguageBind/LanguageBind_Video_merge
git clone https://huggingface.co/LanguageBind/LanguageBind_Image
git clone https://huggingface.co/LanguageBind/LanguageBind_Video

cd ..
echo "prepared checkpoints for OpenCLIP"
mkdir openai
cd openai
git clone https://huggingface.co/openai/clip-vit-large-patch14
git clone https://huggingface.co/openai/clip-vit-base-patch32


cd ..
echo "prepared checkpoints for Vitron-base"
mkdir Vitron-base
cd Vitron-base
git clone https://huggingface.co/Vitron/vitron-base


cd ..
echo "prepared checkpoints for Vitron-lora"
mkdir Vitron-lora
cd Vitron-lora
git clone https://huggingface.co/Vitron/vitron-lora


cd ..
echo "prepared checkpoints for SEEM"
mkdir seem


cd ..
echo "prepared checkpoints for Zeroscope"
mkdir zeroscope
git clone https://huggingface.co/cerspense/zeroscope_v2_576w