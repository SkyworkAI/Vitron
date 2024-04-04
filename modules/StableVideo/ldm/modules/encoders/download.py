from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download('AI-ModelScope/clip-vit-large-patch14', cache_dir='.')
model_dir2 = snapshot_download('zimuwangnlp/flan-t5-xl', cache_dir='.')