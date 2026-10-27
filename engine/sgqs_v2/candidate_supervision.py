"""Training-only geometry used to allocate existing Encoder candidates."""
import torch

from ..core import register
from ..sgqs import SGQSQuerySupervisor


@register()
class SGQSV2Supervisor(SGQSQuerySupervisor):
    """Keep v1 geometric samples; inspect dense boxes without a gradient graph."""

    def __init__(self, scale_priors=(32., 64., 128.), initial_radius=1,
                 max_radius=4, ignore_radius=1, bbox_chunk_size=512):
        super().__init__(scale_priors, initial_radius, max_radius, ignore_radius)
        if int(bbox_chunk_size) != bbox_chunk_size or bbox_chunk_size < 1:
            raise ValueError('bbox_chunk_size must be a positive integer')
        self.bbox_chunk_size = int(bbox_chunk_size)

    def forward(self, output_memory, dense_logits, selection_logits,
                anchors_unact, valid_mask, spatial_shapes, targets, bbox_head,
                feat_strides, num_queries, query_select_method):
        if dense_logits.shape[-1] != 1 or query_select_method == 'one2many':
            raise ValueError('SGQS v2 requires single-class, one-query-per-source selection')
        payload = super().forward(
            output_memory, dense_logits, selection_logits, anchors_unact,
            valid_mask, spatial_shapes, targets, bbox_head, feat_strides,
            num_queries, query_select_method)
        if any(p.dtype != torch.float32 for p in bbox_head.parameters()):
            raise ValueError('SGQS v2 requires FP32 bbox-head weights (AMP is supported)')
        with torch.no_grad(), torch.autocast(output_memory.device.type, enabled=False):
            # Native feature-grid anchors, not the regressed Encoder box center.
            payload['anchor_centers'] = anchors_unact.detach().float()[..., :2].sigmoid().expand(
                output_memory.shape[0], -1, -1)
            boxes = []
            for start in range(0, output_memory.shape[1], self.bbox_chunk_size):
                end = start + self.bbox_chunk_size
                logits = bbox_head(output_memory[:, start:end].detach().float())
                boxes.append((logits + anchors_unact[:, start:end].float()).sigmoid())
            payload['dense_boxes'] = torch.cat(boxes, dim=1)
        return payload
