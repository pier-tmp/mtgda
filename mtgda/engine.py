import torch.nn.functional as F


def encdec_loss(score, step_valid, pick_idx):
    sv = step_valid
    if not sv.any():
        return score.sum() * 0.0
    return F.cross_entropy(score[sv], pick_idx[sv])


def encdec_metrics(score, step_valid, pick_idx):
    top = score.topk(min(3, score.shape[-1]), dim=-1).indices
    sv = step_valid
    n = sv.sum().clamp(min=1)
    top1 = (((top[..., 0] == pick_idx) & sv).sum() / n).item()
    top3 = (((top == pick_idx.unsqueeze(-1)).any(-1) & sv).sum() / n).item()
    return {"top1": top1, "top3": top3}
