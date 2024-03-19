import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

   
class MaskPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):

        if not x.shape[-2:] == mask.shape[-2:]:
            # reshape mask to x
            mask = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)

        b, c, h ,w = x.shape
        b, q, h, w = mask.shape
        mask = (mask > 0).to(mask.dtype)
        denorm = mask.sum(dim=(-1, -2), keepdim=True) + 1e-8

        mask_pooled_x = torch.einsum(
            "bchw,bqhw->bqc",
            x,
            mask / denorm,
        )
        return mask_pooled_x


class LocationEncoder(nn.Module):
    def __init__(self, hidden_size: int):
        super(LocationEncoder, self).__init__()
        self.loc_encoder = nn.Sequential(
            nn.Linear(4, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
    
    def forward(self, x):
        return self.loc_encoder(x)


class RegionExtractor(nn.Module):
    def __init__(self, in_dim=1024, out_dim=4096, patch_size=14, image_size=224):
        """
        Args:
            image_size: the image size of ViT when inputing images
            patch_size: the patch size of ViT when encoding images
        """
        super(RegionExtractor, self).__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        # used for encoding the region features
        self.region_pooling =  MaskPooling()
        self.region_linear = MLP(in_dim, out_dim, out_dim, 3)

        # used for encoding the coordination of bounding box
        self.loc_encoder = LocationEncoder(out_dim)

    def transform_bbox_2_mask(self, bboxes, image_size, device, data_type=torch.float):
        batch_masks = []
        for bbox in bboxes:
            mask = torch.zeros((image_size, image_size), dtype=data_type).to(device)
            x1, y1, x2, y2 = bbox
            # x2, y2 = int(x1 + w), int(y1 + h)
            mask[int(x1):int(x2),int(y1):int(y2)] = 1
            batch_masks.append(mask)
        return torch.stack(batch_masks, dim=0)

    def forward(self, feats, regions):
        """
        To extract the region feartures based on the region mask.
        Args:
            feats(`tensor`): [B, S, H], patch features -> [batch_size, 256, 1024]
            bboxs(`List[List]`): [B, 1, 4], the coordination of the bounding box
        Returns:
            region features: [B, S, H]
            return the region features based on the region mask.
        """
        # print(feats.shape)
        # print(regions.shape)
        b, _, c = feats.shape
        w = h = int(math.sqrt(feats.shape[1]))
        try:
            assert feats.size(0) == len(regions)
        except AssertionError as e:
            print(e)
            print(feats.shape)
            print(len(regions))
            print(regions)
        
        # 1. transform bbox into a region mask with the original image size
        # print('feats.device', feats.device)
        # ori_dtype = feats.dtype
        region_masks = self.transform_bbox_2_mask(regions, self.image_size, feats.device, feats.dtype)
        # print('region_masks: ', region_masks.shape)
        region_masks = region_masks.unsqueeze(1)  # b, 1, image_size, image_size
        # print('region_masks: ', region_masks.shape)
        feats = feats.reshape(b, h, w, c).permute(0, 3, 1, 2)  # b, c, image_size, image_size

        # 2. region_pooling to extract region feautures
        region_feat_raw = self.region_pooling(feats, region_masks)  # b, 1, c, 
        # region_feat_raw = self.region_pooling(feats.to(torch.float), region_masks)  # b, 1, c, 
        # region_feat_raw = region_feat_raw.to(ori_dtype)
        region_feat_flatten = region_feat_raw.reshape(-1, region_feat_raw.shape[-1]) 
        region_feats_linear = self.region_linear(region_feat_flatten)  # b, out_dim

        # 3. encode the cordination of the bbox
        loc_embeds = self.loc_encoder(torch.tensor(regions, dtype=region_feats_linear.dtype, device=region_feats_linear.device)) # b, out_dim

        # 4. concat the region features and loc_embeds
        region_feats = region_feats_linear + loc_embeds
        return region_feats.unsqueeze(1)

    

 
