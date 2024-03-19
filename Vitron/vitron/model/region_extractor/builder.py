from .layer import RegionExtractor

def build_region_extractor(config, delay_load=False, **kwargs):
    # config.mm_hidden_size, config.hidden_size
    return RegionExtractor(config.mm_hidden_size, config.hidden_size)
    