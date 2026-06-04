import torch
import torch.nn.functional as F


def target_in_pack(token_card, pick_batch, cand_idx, target):
    cand_cards = token_card[pick_batch.unsqueeze(1), cand_idx]
    return (cand_cards == target.unsqueeze(1)).float().argmax(dim=1)


def masked_ce(score, cand_mask, target_idx, label_smoothing):
    if label_smoothing == 0.0:
        return F.cross_entropy(score, target_idx)
    logp = F.log_softmax(score, dim=-1)
    logp = logp.masked_fill(cand_mask == 0, 0.0)
    n_valid = cand_mask.sum(dim=-1, keepdim=True)
    eps = label_smoothing
    target = cand_mask.float() / n_valid * eps
    rows = torch.arange(score.shape[0], device=score.device)
    target[rows, target_idx] += 1.0 - eps
    return -(target * logp).sum(dim=-1).mean()


def batch_metrics(score, target_idx):
    top1 = (score.argmax(dim=1) == target_idx).float().mean().item()
    k = min(3, score.shape[1])
    top3 = (score.topk(k, dim=1).indices == target_idx.unsqueeze(1)).any(1).float().mean().item()
    return {"top1": top1, "top3": top3}


def aux_loss(pred, target, mask):
    m = mask.sum()
    if m == 0:
        return pred.sum() * 0.0
    return (((pred - target) ** 2) * mask).sum() / m
