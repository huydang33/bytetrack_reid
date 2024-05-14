# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


"""
Backbone modules.
"""
import torchvision
import torch.nn as nn
import torch
from fastreid.config import get_cfg
from fastreid.modeling import build_model

def build_feature_extract_model(args):
    if args.feat_model_name == 'fastreid':
        assert args.feat_model_weights is not None or args.feat_args_cfg is not None
        
        # Tạo một mô hình mới
        cfg = get_cfg()
        cfg.merge_from_file(args.feat_args_cfg)  # Thay đường dẫn với file cấu hình của mô hình
        model = build_model(cfg)

        # Load trọng số cho mô hình
        checkpoint = torch.load(args.feat_model_weights)

        model.load_state_dict(checkpoint["model"])
    else:
        model = getattr(torchvision.models, args.feat_model_name)(pretrained=True)
        
        # Remove last layer for getting feature map
        model = nn.Sequential(*list(model.children())[:-1])
    return model
