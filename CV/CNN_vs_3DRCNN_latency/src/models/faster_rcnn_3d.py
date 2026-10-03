"""Minimal 3D Faster R-CNN (teaching implementation, pure PyTorch + torchvision).

Architecture: 3D backbone -> 3D RPN -> 3D RoI pooling -> box head.
Backbone: torchvision.video.r3d_18 (Kinetics-pretrained if available),
           features taken before avgpool. Falls back to a tiny 3D CNN if
           torchvision is missing (CPU smoke tests).

Honest scope note: this is NOT a full production detector (single cubic
anchor family, single feature level, slice-crop RoI pooling instead of
trilinear RoIAlign). It IS trainable end-to-end and its memory/latency
profile is representative of the 3D-RPN family vs the 2D slice model —
which is exactly what this benchmark compares.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models.video import r3d_18, R3D_18_Weights
    _HAS_TV = True
except Exception:
    _HAS_TV = False


# ---------------- 3D box utils ----------------

def box_iou_3d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: (N,6), b: (M,6) xyzxyz -> (N,M) IoU."""
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device)
    inter_min = torch.max(a[:, None, :3], b[None, :, :3])
    inter_max = torch.min(a[:, None, 3:], b[None, :, 3:])
    inter = (inter_max - inter_min).clamp(min=0)
    inter_v = inter[..., 0] * inter[..., 1] * inter[..., 2]
    va = ((a[:, 3] - a[:, 0]) * (a[:, 4] - a[:, 1]) * (a[:, 5] - a[:, 2])).clamp(min=1e-6)
    vb = ((b[:, 3] - b[:, 0]) * (b[:, 4] - b[:, 1]) * (b[:, 5] - b[:, 2])).clamp(min=1e-6)
    return inter_v / (va[:, None] + vb[None, :] - inter_v + 1e-6)


def nms_3d(boxes: torch.Tensor, scores: torch.Tensor, thresh: float = 0.4, topk: int = 100) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)[:topk]
    keep = []
    while order.numel():
        i = order[0]
        keep.append(i.item())
        if order.numel() == 1:
            break
        ious = box_iou_3d(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious < thresh]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def encode_boxes(anchors: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """FasterRCNN-style deltas (dc, dw log) in xyz order. (N,6)."""
    ac = (anchors[:, :3] + anchors[:, 3:]) / 2
    aw = (anchors[:, 3:] - anchors[:, :3]).clamp(min=1e-3)
    gc = (gt[:, :3] + gt[:, 3:]) / 2
    gw = (gt[:, 3:] - gt[:, :3]).clamp(min=1e-3)
    return torch.cat([(gc - ac) / aw, torch.log(gw / aw)], dim=1)


def decode_boxes(anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    ac = (anchors[:, :3] + anchors[:, 3:]) / 2
    aw = (anchors[:, 3:] - anchors[:, :3]).clamp(min=1e-3)
    dc, dw = deltas[:, :3], deltas[:, 3:]
    gc = dc * aw + ac
    gw = torch.exp(dw.clamp(-4, 4)) * aw
    return torch.cat([gc - gw / 2, gc + gw / 2], dim=1)


# ---------------- backbone ----------------

class TinyBackbone3D(nn.Module):
    """Fallback when torchvision is unavailable. Stride 8, 128 ch."""

    def __init__(self):
        super().__init__()
        self.out_channels = 128
        self.stride = 8
        c = [16, 32, 64, 128]
        layers = []
        in_c = 1
        for i, o in enumerate(c):
            s = (2, 2, 2) if i < 3 else (1, 1, 1)
            layers += [nn.Conv3d(in_c, o, 3, stride=s, padding=1), nn.BatchNorm3d(o), nn.ReLU(True)]
            in_c = o
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # 1,D/8,H/8,W/8 order? conv3d is (N,C,D,H,W) ✓


def _to_single_channel_3d(feats: nn.Module) -> nn.Module:
    """Convert the first 3-channel Conv3d (Kinetics RGB stem) to 1-channel CT.

    Seeds the new kernel with the channel-mean of the pretrained weights,
    preserving the pretrained filters. No-op if no 3-channel conv exists.
    """
    for name, mod in feats.named_modules():
        if isinstance(mod, nn.Conv3d) and mod.in_channels == 3:
            w = mod.weight.data.mean(dim=1, keepdim=True)
            new = nn.Conv3d(1, mod.out_channels, kernel_size=mod.kernel_size,
                            stride=mod.stride, padding=mod.padding,
                            dilation=mod.dilation, bias=mod.bias is not None)
            new.weight.data.copy_(w)
            if mod.bias is not None and new.bias is not None:
                new.bias.data.copy_(mod.bias.data)
            parent = feats
            *path, leaf = name.split(".")
            for p in path:
                parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
            if leaf.isdigit():
                parent[int(leaf)] = new
            else:
                setattr(parent, leaf, new)
            break
    return feats


def build_backbone_3d(pretrained: bool = True):
    if _HAS_TV:
        try:
            w = R3D_18_Weights.KINETICS400_V1 if pretrained else None
            m = r3d_18(weights=w)
            feats = nn.Sequential(m.stem, m.layer1, m.layer2, m.layer3)  # stride 16
            return _to_single_channel_3d(feats), 256, 16
        except Exception:
            pass
    return TinyBackbone3D(), 128, 8


# ---------------- model ----------------

class FasterRCNN3D(nn.Module):
    def __init__(
        self,
        anchor_sizes=(16, 32, 64),
        rpn_pre_nms: int = 1000,
        rpn_post_nms: int = 256,
        roi_size=(7, 7, 7),
        box_score_thresh: float = 0.3,
        nms_thresh: float = 0.4,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.backbone, feat_ch, self.stride = build_backbone_3d(pretrained_backbone)
        self.anchor_sizes = list(anchor_sizes)
        self.rpn_pre, self.rpn_post = rpn_pre_nms, rpn_post_nms
        self.roi_size = tuple(roi_size)
        self.score_thresh, self.nms_thresh = box_score_thresh, nms_thresh

        self.rpn_conv = nn.Conv3d(feat_ch, feat_ch, 3, padding=1)
        self.rpn_cls = nn.Conv3d(feat_ch, len(self.anchor_sizes), 1)   # objectness logit
        self.rpn_reg = nn.Conv3d(feat_ch, len(self.anchor_sizes) * 6, 1)

        rd, rh, rw = self.roi_size
        self.head_fc = nn.Sequential(nn.Flatten(), nn.Linear(feat_ch * rd * rh * rw, 512), nn.ReLU(True))
        self.cls_head = nn.Linear(512, 2)   # bg / lesion
        self.reg_head = nn.Linear(512, 6)   # class-agnostic deltas

    # -- anchors --
    def _anchors_for(self, feat_shape, vol_shape, device) -> torch.Tensor:
        fd, fh, fw = feat_shape
        D, H, W = vol_shape
        sz = torch.tensor(self.anchor_sizes, device=device).float()
        zz = (torch.arange(fd, device=device) + 0.5) * self.stride
        yy = (torch.arange(fh, device=device) + 0.5) * self.stride
        xx = (torch.arange(fw, device=device) + 0.5) * self.stride
        gz, gy, gx = torch.meshgrid(zz, yy, xx, indexing="ij")
        ctr = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)  # (K,3) xyz
        K = ctr.shape[0]
        A = len(sz)
        half = sz / 2  # (A,)
        boxes = torch.empty((K * A, 6), device=device)
        for a in range(A):
            boxes[a::A, :3] = ctr - half[a]
            boxes[a::A, 3:] = ctr + half[a]
        boxes[:, [0, 3]] = boxes[:, [0, 3]].clamp(0, W)
        boxes[:, [1, 4]] = boxes[:, [1, 4]].clamp(0, H)
        boxes[:, [2, 5]] = boxes[:, [2, 5]].clamp(0, D)
        return boxes

    def _roi_pool(self, feat: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """Crop + adaptive pool per box on the feature map. feat: (C,fd,fh,fw)."""
        C = feat.shape[0]
        out = []
        for b in boxes:
            x1, y1, z1, x2, y2, z2 = (b / self.stride).long().tolist()
            fd, fh, fw = feat.shape[1:]
            x1, x2 = max(x1, 0), min(max(x2, x1 + 1), fw)
            y1, y2 = max(y1, 0), min(max(y2, y1 + 1), fh)
            z1, z2 = max(z1, 0), min(max(z2, z1 + 1), fd)
            crop = feat[:, z1:z2, y1:y2, x1:x2]
            out.append(F.adaptive_max_pool3d(crop[None], self.roi_size)[0])
        if not out:
            return torch.empty((0, C, *self.roi_size), device=feat.device)
        return torch.stack(out)

    # -- forward --
    def forward(self, volumes: torch.Tensor, targets=None):
        """volumes: (B,1,D,H,W). targets: list of {boxes, labels} in VOXEL xyzxyz."""
        B, _, D, H, W = volumes.shape
        feats = self.backbone(volumes)  # (B,C,fd,fh,fw)
        fd, fh, fw = feats.shape[2:]
        device = volumes.device

        rpn_logit = self.rpn_cls(F.relu(self.rpn_conv(feats)))     # (B,A,fd,fh,fw)
        rpn_delta = self.rpn_reg(F.relu(self.rpn_conv(feats)))
        rpn_logit_f = rpn_logit.permute(0, 2, 3, 4, 1).reshape(B, -1)
        rpn_delta_f = rpn_delta.permute(0, 2, 3, 4, 1).reshape(B, -1, 6)

        anchors = self._anchors_for((fd, fh, fw), (D, H, W), device)
        if self.training and targets is not None:
            return self._forward_train(feats, anchors, rpn_logit_f, rpn_delta_f, targets)
        return self._forward_infer(feats, anchors, rpn_logit_f, rpn_delta_f, (D, H, W))

    def _forward_train(self, feats, anchors, logit_f, delta_f, targets):
        B = feats.shape[0]
        loss_cls = loss_reg = loss_det_cls = loss_det_reg = torch.tensor(0.0, device=feats.device)
        for b in range(B):
            gt = targets[b]["boxes"].to(feats.device)
            if gt.numel() == 0:  # no lesion: everything is background
                loss_cls = loss_cls + F.binary_cross_entropy_with_logits(
                    logit_f[b], torch.zeros_like(logit_f[b]))
                continue
            iou = box_iou_3d(anchors, gt)
            max_iou, gt_idx = iou.max(dim=1)
            pos = max_iou >= 0.5
            neg = max_iou < 0.3
            # subsample to 128 (pos:neg ~ 1:1)
            n_pos = int(min(pos.sum(), 64))
            n_neg = int(min(neg.sum(), 128 - n_pos))
            p_idx = torch.where(pos)[0][:n_pos]
            n_idx = torch.where(neg)[0][:n_neg]
            if len(p_idx) == 0:  # degenerate: skip reg, still learn bg
                tgt = torch.zeros_like(logit_f[b])
                loss_cls = loss_cls + F.binary_cross_entropy_with_logits(logit_f[b], tgt)
                continue
            tgt = torch.zeros_like(logit_f[b])
            tgt[p_idx] = 1.0
            keep = torch.cat([p_idx, n_idx])
            loss_cls = loss_cls + F.binary_cross_entropy_with_logits(logit_f[b][keep], tgt[keep])
            loss_reg = loss_reg + F.smooth_l1_loss(
                delta_f[b][p_idx], encode_boxes(anchors[p_idx], gt[gt_idx[p_idx]]))

            # detector head on jittered GT + sampled anchors (simplified "proposals")
            prop = torch.cat([gt, anchors[p_idx][: min(32, len(p_idx))]])
            pooled = self._roi_pool(feats[b], prop)
            h = self.head_fc(pooled)
            cls_l = self.cls_head(h)
            reg_l = self.reg_head(h)
            dlab = torch.cat([torch.ones(len(gt), device=h.device), torch.zeros(len(prop) - len(gt), device=h.device)]).long()
            loss_det_cls = loss_det_cls + F.cross_entropy(cls_l, dlab)
            if len(gt):
                loss_det_reg = loss_det_reg + F.smooth_l1_loss(
                    reg_l[: len(gt)], encode_boxes(prop[: len(gt)], gt))
        n = max(B, 1)
        return {"loss_rpn_cls": loss_cls / n, "loss_rpn_reg": loss_reg / n,
                "loss_det_cls": loss_det_cls / n, "loss_det_reg": loss_det_reg / n}

    @torch.no_grad()
    def _forward_infer(self, feats, anchors, logit_f, delta_f, vol_shape):
        B = feats.shape[0]
        D, H, W = vol_shape
        results = []
        for b in range(B):
            scores = logit_f[b].sigmoid()
            k = min(self.rpn_pre, scores.numel())
            top = scores.topk(k).indices
            prop = decode_boxes(anchors[top], delta_f[b][top])
            prop[:, [0, 3]] = prop[:, [0, 3]].clamp(0, W)
            prop[:, [1, 4]] = prop[:, [1, 4]].clamp(0, H)
            prop[:, [2, 5]] = prop[:, [2, 5]].clamp(0, D)
            keep = nms_3d(prop, scores[top], self.nms_thresh, self.rpn_post)
            prop = prop[keep]
            if len(prop) == 0:
                results.append({"boxes": torch.zeros((0, 6)), "scores": torch.zeros((0,))})
                continue
            pooled = self._roi_pool(feats[b], prop.to(feats.device))
            h = self.head_fc(pooled)
            cls = self.cls_head(h).softmax(-1)[:, 1]
            reg = self.reg_head(h)
            boxes = decode_boxes(prop.to(feats.device), reg)
            m = cls > self.score_thresh
            boxes, cls = boxes[m], cls[m]
            keep2 = nms_3d(boxes, cls, self.nms_thresh)
            results.append({"boxes": boxes[keep2].cpu(), "scores": cls[keep2].cpu()})
        return results
