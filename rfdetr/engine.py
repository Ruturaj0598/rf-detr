# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable
import random

import torch
import torch.nn.functional as F

import rfdetr.util.misc as utils
from rfdetr.datasets.coco_eval import CocoEvaluator
from rfdetr.datasets.coco import compute_multi_scale_scales
from rfdetr.util.cka_loss import (
    MultiLayerCKALoss,
    EnhancedAugmentation,
    PairwiseMultiLayerCKA
)

try:
    from torch.amp import autocast, GradScaler
    DEPRECATED_AMP = False
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    DEPRECATED_AMP = True
from typing import DefaultDict, List, Callable
from rfdetr.util.misc import NestedTensor
import numpy as np

# ============ CKA CONTRASTIVE LOSS HELPERS ============

def augment_view_batch(batch: torch.Tensor, p_hflip=0.5, scale_min=0.9, scale_max=1.1):
    """
    Apply simple augmentations to create a second view for CKA loss.
    
    Args:
        batch: Input tensor [B, 3, H, W] normalized to [0, 1]
        p_hflip: Probability of horizontal flip
        scale_min: Minimum scale factor
        scale_max: Maximum scale factor
    
    Returns:
        Augmented batch tensor
    """
    # Random horizontal flip
    if torch.rand(1).item() < p_hflip:
        batch = torch.flip(batch, dims=[3])  # flip width dim
    
    # Random scale/resample (center crop/rescale)
    _, _, H, W = batch.shape
    scale = float(torch.empty(1).uniform_(scale_min, scale_max).item())
    new_h = max(1, int(H * scale))
    new_w = max(1, int(W * scale))
    
    # bilinear resize then center-crop/pad back to original
    batch_resized = F.interpolate(batch, size=(new_h, new_w), mode='bilinear', align_corners=False)
    
    # center crop or pad back to H,W
    if new_h >= H and new_w >= W:
        top = (new_h - H) // 2
        left = (new_w - W) // 2
        batch_aug = batch_resized[..., top:top+H, left:left+W]
    else:
        # pad
        pad_h = max(0, H - new_h)
        pad_w = max(0, W - new_w)
        pad = (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2)
        batch_aug = F.pad(batch_resized, pad, mode='reflect')
        batch_aug = batch_aug[..., :H, :W]
    
    return batch_aug.clamp(0.0, 1.0)


def linear_cka_from_feature_maps(feat_a: torch.Tensor, feat_b: torch.Tensor, eps=1e-5):
    """
    Compute linear CKA similarity between two feature maps.
    
    Args:
        feat_a, feat_b: Feature tensors [B, C, H, W]
        eps: Small epsilon for numerical stability
    
    Returns:
        Per-sample CKA values [B]
    """
    B, Ca, Ha, Wa = feat_a.shape
    B2, Cb, Hb, Wb = feat_b.shape
    assert B == B2 and Ha == Hb and Wa == Wb, "feature maps must match spatial size and batch"
    n = Ha * Wa
    
    # reshape to [B, n, C]
    A = feat_a.view(B, Ca, n).permute(0, 2, 1).contiguous()
    Bf = feat_b.view(B, Cb, n).permute(0, 2, 1).contiguous()
    
    # center along samples (n)
    A = A - A.mean(dim=1, keepdim=True)
    Bf = Bf - Bf.mean(dim=1, keepdim=True)
    
    # compute cross-covariance matrices
    AtB = torch.bmm(A.transpose(1, 2), Bf)  # [B, Ca, Cb]
    
    # HSIC_AB = sum of squares of AtB entries
    hsic_ab = (AtB ** 2).view(B, -1).sum(dim=1)
    
    AtA = torch.bmm(A.transpose(1, 2), A)  # [B, Ca, Ca]
    Btb = torch.bmm(Bf.transpose(1, 2), Bf)  # [B, Cb, Cb]
    
    hsic_aa = (AtA ** 2).view(B, -1).sum(dim=1)
    hsic_bb = (Btb ** 2).view(B, -1).sum(dim=1)
    
    denom = torch.sqrt(hsic_aa * hsic_bb)
    cka = hsic_ab / (denom + eps)
    
    # clamp to [0,1]
    cka = torch.clamp(cka, 0.0, 1.0)
    return cka

# ============ END CKA HELPERS ============

def get_autocast_args(args):
    if DEPRECATED_AMP:
        return {'enabled': args.amp, 'dtype': torch.bfloat16}
    else:
        return {'device_type': 'cuda', 'enabled': args.amp, 'dtype': torch.bfloat16}

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    batch_size: int,
    max_norm: float = 0,
    ema_m: torch.nn.Module = None,
    schedules: dict = {},
    num_training_steps_per_epoch=None,
    vit_encoder_num_layers=None,
    args=None,
    callbacks: DefaultDict[str, List[Callable]] = None,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10
    start_steps = epoch * num_training_steps_per_epoch

        # ============================================================
    # INITIALIZE MULTI-LAYER CKA LOSS
    # ============================================================
    multilayer_cka_loss = None
    if getattr(args, 'cka_lambda', 0.0) > 0:
        # Configure layer weights (emphasize earlier layers for small objects)
        layer_weights = None
        if getattr(args, 'cka_focus_small_objects', True):
            layer_weights = {
                0: 2.0,  # Highest resolution - most important for small objects
                1: 1.5,  # Medium resolution
                2: 1.0   # Lowest resolution (if exists)
            }
        
        multilayer_cka_loss = MultiLayerCKALoss(
            layer_weights=layer_weights,
            progressive_schedule=getattr(args, 'cka_progressive', False),
            focus_on_small_objects=getattr(args, 'cka_focus_small_objects', True),
            temperature=getattr(args, 'cka_temperature', 1.0)
        )
        
        # Set current epoch for progressive weighting
        multilayer_cka_loss.set_epoch(epoch)
        
        print(f"\n{'='*60}")
        print(f"Multi-Layer CKA Loss Configuration (Epoch {epoch})")
        print(f"{'='*60}")
        print(f"  Lambda (weight): {args.cka_lambda}")
        print(f"  Augmentation strength: {getattr(args, 'cka_augmentation_strength', 'moderate')}")
        print(f"  Progressive weighting: {getattr(args, 'cka_progressive', False)}")
        if getattr(args, 'cka_progressive', False):
            progressive_weight = multilayer_cka_loss.get_progressive_weight()
            print(f"    Current progressive weight: {progressive_weight:.3f}")
        print(f"  Small object focus: {getattr(args, 'cka_focus_small_objects', True)}")
        if layer_weights:
            print(f"    Layer weights: {layer_weights}")
        print(f"  Temperature: {getattr(args, 'cka_temperature', 1.0)}")
        print(f"{'='*60}\n")
    # ============================================================

    print("Grad accum steps: ", args.grad_accum_steps)
    print("Total batch size: ", batch_size * utils.get_world_size())

    # Add gradient scaler for AMP
    if DEPRECATED_AMP:
        scaler = GradScaler(enabled=args.amp)
    else:
        scaler = GradScaler('cuda', enabled=args.amp)

    optimizer.zero_grad()
    assert batch_size % args.grad_accum_steps == 0
    sub_batch_size = batch_size // args.grad_accum_steps
    print("LENGTH OF DATA LOADER:", len(data_loader))
    
    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        it = start_steps + data_iter_step
        callback_dict = {
            "step": it,
            "model": model,
            "epoch": epoch,
        }
        for callback in callbacks["on_train_batch_start"]:
            callback(callback_dict)
        if "dp" in schedules:
            if args.distributed:
                model.module.update_drop_path(
                    schedules["dp"][it], vit_encoder_num_layers
                )
            else:
                model.update_drop_path(schedules["dp"][it], vit_encoder_num_layers)
        if "do" in schedules:
            if args.distributed:
                model.module.update_dropout(schedules["do"][it])
            else:
                model.update_dropout(schedules["do"][it])

        if args.multi_scale and not args.do_random_resize_via_padding:
            scales = compute_multi_scale_scales(args.resolution, args.expanded_scales, args.patch_size, args.num_windows)
            random.seed(it)
            scale = random.choice(scales)
            with torch.inference_mode():
                samples.tensors = F.interpolate(samples.tensors, size=scale, mode='bilinear', align_corners=False)
                samples.mask = F.interpolate(samples.mask.unsqueeze(1).float(), size=scale, mode='nearest').squeeze(1).bool()

        for i in range(args.grad_accum_steps):
            start_idx = i * sub_batch_size
            final_idx = start_idx + sub_batch_size
            new_samples_tensors = samples.tensors[start_idx:final_idx]
            new_samples = NestedTensor(new_samples_tensors, samples.mask[start_idx:final_idx])
            new_samples = new_samples.to(device)
            new_targets = [{k: v.to(device) for k, v in t.items()} for t in targets[start_idx:final_idx]]
            
            # ============ ACCESS MODEL CORRECTLY ============
            if hasattr(model, 'module'):
                internal_net = model.module  # DDP wrapped
            else:
                internal_net = model  # Direct LWDETR instance
            # ================================================

            with autocast(**get_autocast_args(args)):
                # Forward pass for view A
                outputs_a = internal_net(new_samples)
                loss_dict = criterion(outputs_a, new_targets)
                weight_dict = criterion.weight_dict

                # ============================================================
                # MULTI-LAYER CKA REGULARIZATION
                # ============================================================
                if multilayer_cka_loss is not None:
                    # Get augmentation strength from args
                    aug_strength = getattr(args, 'cka_augmentation_strength', 'moderate')
                    
                    # Generate enhanced augmented view B using diverse augmentations
                    view_b_tensors = EnhancedAugmentation.create_augmented_view(
                        new_samples.tensors,
                        augmentation_strength=aug_strength
                    )
                    view_b = NestedTensor(view_b_tensors, new_samples.mask)

                    # Forward pass for view B (no gradients needed)
                    with torch.no_grad():
                        outputs_b = internal_net(view_b)

                    # Extract multi-layer features from both views
                    features_a = outputs_a.get("backbone_features", {})
                    features_b = outputs_b.get("backbone_features", {})

                    if features_a and features_b:
                        # Compute multi-layer CKA loss with hierarchical weighting
                        cka_loss, layer_losses = multilayer_cka_loss(features_a, features_b)
                        
                        # Add weighted CKA loss to total loss
                        loss_dict["loss_cka"] = cka_loss * args.cka_lambda
                        
                        # Add individual layer losses for detailed logging
                        for key, value in layer_losses.items():
                            if isinstance(value, (int, float)):
                                loss_dict[f"cka_{key}"] = torch.tensor(value, device=device)
                        
                        # Log layer-wise information (first iteration only to avoid spam)
                        if i == 0 and data_iter_step % print_freq == 0:
                            print(f"\n  [CKA Layer Losses - Step {data_iter_step}]")
                            for layer_idx in sorted([k for k in features_a.keys()]):
                                layer_loss = layer_losses.get(f'cka_layer_{layer_idx}', 0)
                                print(f"    Layer {layer_idx}: {layer_loss:.6f}")
                            print(f"    Total CKA: {layer_losses.get('cka_total', 0):.6f}")
                            print(f"    Progressive weight: {layer_losses.get('progressive_weight', 1.0):.3f}\n")
                    else:
                        # Fallback if features not available
                        loss_dict["loss_cka"] = torch.tensor(0.0, device=device)
                        if i == 0 and data_iter_step % (print_freq * 5) == 0:
                            print(f"  ⚠️ Warning: No backbone features found for CKA computation")
                # ============================================================
                # END MULTI-LAYER CKA REGULARIZATION
                # ============================================================

                # Compute aggregated losses
                losses = sum(
                    (1 / args.grad_accum_steps) * loss_dict[k] * weight_dict[k]
                    for k in loss_dict.keys()
                    if k in weight_dict
                )
                
                # Add CKA loss if not in weight_dict
                if 'loss_cka' in loss_dict and 'loss_cka' not in weight_dict:
                    losses = losses + (1 / args.grad_accum_steps) * loss_dict['loss_cka']

            scaler.scale(losses).backward()


        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k:  v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        
        # ============ ADD CKA LOSS TO SCALED DICT ============
        if 'loss_cka' in loss_dict_reduced and 'loss_cka' not in weight_dict:
            loss_dict_reduced_scaled['loss_cka'] = loss_dict_reduced['loss_cka'] * getattr(args, 'cka_lambda', 0.0)
        # =====================================================
        
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()


        if not math.isfinite(loss_value):
            print(loss_dict_reduced)
            raise ValueError("Loss is {}, stopping training".format(loss_value))

        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad()
        if ema_m is not None:
            if epoch >= 0:
                ema_m.update(model)
        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def coco_extended_metrics(coco_eval):
    """
    Safe version: ignores the –1 sentinel entries so precision/F1 never explode.
    """

    iou_thrs, rec_thrs = coco_eval.params.iouThrs, coco_eval.params.recThrs
    iou50_idx, area_idx, maxdet_idx = (
        int(np.argwhere(np.isclose(iou_thrs, 0.50))), 0, 2)

    P = coco_eval.eval["precision"]
    S = coco_eval.eval["scores"]

    prec_raw = P[iou50_idx, :, :, area_idx, maxdet_idx]

    prec = prec_raw.copy().astype(float)
    prec[prec < 0] = np.nan

    f1_cls   = 2 * prec * rec_thrs[:, None] / (prec + rec_thrs[:, None])
    f1_macro = np.nanmean(f1_cls, axis=1)

    best_j   = int(f1_macro.argmax())

    macro_precision = float(np.nanmean(prec[best_j]))
    macro_recall    = float(rec_thrs[best_j])
    macro_f1        = float(f1_macro[best_j])

    score_vec = S[iou50_idx, best_j, :, area_idx, maxdet_idx].astype(float)
    score_vec[prec_raw[best_j] < 0] = np.nan
    score_thr = float(np.nanmean(score_vec))

    map_50_95, map_50 = float(coco_eval.stats[0]), float(coco_eval.stats[1])

    per_class = []
    cat_ids = coco_eval.params.catIds
    cat_id_to_name = {c["id"]: c["name"] for c in coco_eval.cocoGt.loadCats(cat_ids)}
    for k, cid in enumerate(cat_ids):
        p_slice = P[:, :, k, area_idx, maxdet_idx]
        valid   = p_slice > -1
        ap_50_95 = float(p_slice[valid].mean()) if valid.any() else float("nan")
        ap_50    = float(p_slice[iou50_idx][p_slice[iou50_idx] > -1].mean()) if (p_slice[iou50_idx] > -1).any() else float("nan")

        pc = float(prec[best_j, k]) if prec_raw[best_j, k] > -1 else float("nan")
        rc = macro_recall

        #Doing to this to filter out dataset class
        if np.isnan(ap_50_95) or np.isnan(ap_50) or np.isnan(pc) or np.isnan(rc):
            continue

        per_class.append({
            "class"      : cat_id_to_name[int(cid)],
            "map@50:95"  : ap_50_95,
            "map@50"     : ap_50,
            "precision"  : pc,
            "recall"     : rc,
        })

    per_class.append({
        "class"     : "all",
        "map@50:95" : map_50_95,
        "map@50"    : map_50,
        "precision" : macro_precision,
        "recall"    : macro_recall,
    })

    return {
        "class_map": per_class,
        "map"      : map_50,
        "precision": macro_precision,
        "recall"   : macro_recall
    }


def evaluate(model, criterion, postprocess, data_loader, base_ds, device, args=None):
    model.eval()
    if args.fp16_eval:
        model.half()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Test:"

    iou_types = ("bbox",) if not args.segmentation_head else ("bbox", "segm")
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if args.fp16_eval:
            samples.tensors = samples.tensors.half()

        # Add autocast for evaluation
        with autocast(**get_autocast_args(args)):
            outputs = model(samples)

        if args.fp16_eval:
            for key in outputs.keys():
                if key == "enc_outputs":
                    for sub_key in outputs[key].keys():
                        outputs[key][sub_key] = outputs[key][sub_key].float()
                elif key == "aux_outputs":
                    for idx in range(len(outputs[key])):
                        for sub_key in outputs[key][idx].keys():
                            outputs[key][idx][sub_key] = outputs[key][idx][
                                sub_key
                            ].float()
                else:
                    outputs[key] = outputs[key].float()

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results_all = postprocess(outputs, orig_target_sizes)
        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results_all)
        }
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        results_json = coco_extended_metrics(coco_evaluator.coco_eval["bbox"])
        stats["results_json"] = results_json
        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()

        if "segm" in iou_types:
            results_json = coco_extended_metrics(coco_evaluator.coco_eval["segm"])
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
    return stats, coco_evaluator
