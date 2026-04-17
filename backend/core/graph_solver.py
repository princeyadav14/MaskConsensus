from typing import Dict, List, Tuple, Optional
import math

import numpy as np
import torch
import torch.nn.functional as F


EPS = 1e-8


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _normalize_embeddings(embs: List[torch.Tensor]) -> torch.Tensor:
    x = torch.stack([e.flatten().detach().cpu().float() for e in embs], dim=0)
    return F.normalize(x, p=2, dim=1, eps=EPS)


def _legacy_embedding_only_consensus(embeddings_list: List[List[torch.Tensor]]) -> Dict:
    """
    Backward-compatible fallback if mask metadata is not provided.
    This is intentionally simple.
    """
    n_images = len(embeddings_list)
    if n_images == 0:
        return {
            "winning_indices": [],
            "winning_scores": [],
            "global_affinity": 0.0,
            "nodes_evaluated": 0,
            "solver_name": "legacy_embedding_only",
        }
    if n_images == 1:
        return {
            "winning_indices": [0],
            "winning_scores": [1.0],
            "global_affinity": 1.0,
            "nodes_evaluated": 0,
            "solver_name": "legacy_embedding_only",
        }
    if any(len(embs) == 0 for embs in embeddings_list):
        return {
            "winning_indices": [0] * n_images,
            "winning_scores": [0.0] * n_images,
            "global_affinity": 0.0,
            "nodes_evaluated": 0,
            "solver_name": "legacy_embedding_only",
        }

    normalized = [_normalize_embeddings(embs) for embs in embeddings_list]
    n_masks_per_image = [x.shape[0] for x in normalized]
    consensus_scores = np.zeros((n_images, max(n_masks_per_image)), dtype=np.float32)
    nodes_evaluated = 0

    for i in range(n_images):
        for mask_idx_i in range(n_masks_per_image[i]):
            emb_i = normalized[i][mask_idx_i]
            best_scores = []
            for j in range(n_images):
                if i == j:
                    continue
                sims = torch.mv(normalized[j], emb_i).clamp(min=0.0, max=1.0)
                nodes_evaluated += int(sims.numel())
                best_scores.append(float(sims.max().item()))
            consensus_scores[i, mask_idx_i] = float(np.mean(best_scores))

    winning_indices = [int(np.argmax(consensus_scores[i, : n_masks_per_image[i]])) for i in range(n_images)]
    winning_scores = [float(consensus_scores[i, winning_indices[i]]) for i in range(n_images)]
    global_affinity = float(np.mean(winning_scores)) if winning_scores else 0.0

    return {
        "winning_indices": winning_indices,
        "winning_scores": winning_scores,
        "global_affinity": global_affinity,
        "nodes_evaluated": nodes_evaluated,
        "solver_name": "legacy_embedding_only",
    }


def _extract_meta(mask_info: Dict, image_shape: Tuple[int, int]) -> Dict:
    h, w = image_shape
    x1, y1, x2, y2 = mask_info["bbox"]

    bbox_w = max(1, int(x2) - int(x1) + 1)
    bbox_h = max(1, int(y2) - int(y1) + 1)
    bbox_area = bbox_w * bbox_h
    image_area = float(h * w)

    area = float(mask_info["area"])
    area_ratio = area / max(image_area, 1.0)
    bbox_area_ratio = bbox_area / max(image_area, 1.0)

    cx = (x1 + x2 + 1) / 2.0 / max(w, 1)
    cy = (y1 + y2 + 1) / 2.0 / max(h, 1)

    # normalized distance from image center
    center_dist = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2) / math.sqrt(0.5**2 + 0.5**2)
    center_dist = float(np.clip(center_dist, 0.0, 1.0))

    touches_border = int(x1 <= 1 or y1 <= 1 or x2 >= w - 2 or y2 >= h - 2)
    sam_score = float(mask_info.get("score", 0.5))

    return {
        "area_ratio": float(area_ratio),
        "bbox_area_ratio": float(bbox_area_ratio),
        "center_dist": center_dist,
        "touches_border": touches_border,
        "sam_score": float(np.clip(sam_score, 0.0, 1.0)),
    }


def _candidate_is_valid(meta: Dict) -> bool:
    """
    Hard filter for obviously bad fragment masks.
    We only exclude very tiny or tiny-border masks.
    """
    area_ratio = meta["area_ratio"]
    bbox_area_ratio = meta["bbox_area_ratio"]
    touches_border = meta["touches_border"]

    if area_ratio < 0.0010:
        return False
    if bbox_area_ratio < 0.0015:
        return False
    if touches_border and area_ratio < 0.025:
        return False
    return True


def _area_prior(area_ratio: float) -> float:
    """
    Reward masks that are not tiny and not nearly full-image.
    Broad enough to allow varying object sizes.
    """
    a = max(area_ratio, EPS)

    small_gate = _sigmoid((a - 0.015) / 0.010)
    huge_gate = _sigmoid((0.90 - a) / 0.08)

    # broad peak around moderate object sizes
    mu = math.log(0.18)
    sigma = 0.95
    z = (math.log(a) - mu) / sigma
    peak = math.exp(-0.5 * z * z)

    score = small_gate * huge_gate * peak
    return float(np.clip(score, 0.0, 1.0))


def _center_prior(center_dist: float) -> float:
    return float(math.exp(-3.0 * (center_dist ** 2)))


def _border_fragment_penalty(meta: Dict) -> float:
    a = meta["area_ratio"]
    ba = meta["bbox_area_ratio"]
    touches = meta["touches_border"]

    small_border = touches * _sigmoid((0.030 - a) / 0.008)
    tiny_bbox = _sigmoid((0.015 - ba) / 0.004)
    return float(np.clip(0.70 * small_border + 0.30 * tiny_bbox, 0.0, 1.0))


def _unary_score(meta: Dict) -> float:
    """
    Single-mask plausibility score:
    good SAM score + reasonable size + mild center preference - border-fragment penalty
    """
    score_prior = meta["sam_score"]
    area_prior = _area_prior(meta["area_ratio"])
    center_prior = _center_prior(meta["center_dist"])
    border_pen = _border_fragment_penalty(meta)

    value = (
        0.50 * score_prior
        + 0.30 * area_prior
        + 0.10 * center_prior
        - 0.35 * border_pen
    )
    return float(value)


def _log_ratio_compat(x: torch.Tensor, y: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.exp(-torch.abs(torch.log((x + EPS) / (y + EPS))) / tau)


def _build_pair_affinity(
    A: torch.Tensor,
    B: torch.Tensor,
    meta_i: List[Dict],
    meta_j: List[Dict],
    mutual_bonus: float = 0.05,
) -> Tuple[torch.Tensor, float, int]:
    """
    Pairwise affinity between masks of image i and image j.
    Combines semantic embedding similarity with geometric compatibility.
    """
    sim = (A @ B.T).clamp(min=0.0, max=1.0)
    nodes_evaluated = int(sim.numel())

    ai = torch.tensor([m["area_ratio"] for m in meta_i], dtype=torch.float32).unsqueeze(1)
    aj = torch.tensor([m["area_ratio"] for m in meta_j], dtype=torch.float32).unsqueeze(0)
    area_compat = _log_ratio_compat(ai, aj, tau=1.25)

    si = torch.tensor([m["sam_score"] for m in meta_i], dtype=torch.float32).unsqueeze(1)
    sj = torch.tensor([m["sam_score"] for m in meta_j], dtype=torch.float32).unsqueeze(0)
    score_compat = torch.exp(-torch.abs(si - sj) / 0.35)

    # Main signal is semantic similarity, but we modulate with size and SAM score compatibility.
    affinity = sim * (0.70 + 0.30 * area_compat) * (0.85 + 0.15 * score_compat)
    affinity = affinity.clamp(min=0.0, max=1.0)

    # Mutual nearest-neighbor bonus
    if affinity.numel() > 0:
        row_best_idx = torch.argmax(affinity, dim=1)
        col_best_idx = torch.argmax(affinity, dim=0)
        for i in range(affinity.shape[0]):
            j = int(row_best_idx[i].item())
            if int(col_best_idx[j].item()) == i:
                affinity[i, j] = min(1.0, float(affinity[i, j].item()) + mutual_bonus)

    row_best = affinity.max(dim=1).values
    col_best = affinity.max(dim=0).values
    pair_weight = 0.5 * (float(row_best.mean().item()) + float(col_best.mean().item()))
    pair_weight = float(np.clip(pair_weight, 0.05, 1.0))

    return affinity, pair_weight, nodes_evaluated


def _compute_unary_scores(
    metas: List[List[Dict]],
) -> List[torch.Tensor]:
    unary = []
    for meta_list in metas:
        scores = [_unary_score(m) for m in meta_list]
        unary.append(torch.tensor(scores, dtype=torch.float32))
    return unary


def _objective(
    selection: List[int],
    unary: List[torch.Tensor],
    affinity: Dict[Tuple[int, int], torch.Tensor],
    pair_weights: np.ndarray,
    pair_lambda: float,
) -> float:
    n_images = len(selection)
    obj = 0.0

    for i in range(n_images):
        obj += float(unary[i][selection[i]].item())

    for i in range(n_images):
        for j in range(i + 1, n_images):
            obj += pair_lambda * float(pair_weights[i, j]) * float(
                affinity[(i, j)][selection[i], selection[j]].item()
            )

    return obj


def _coordinate_ascent(
    init_selection: List[int],
    unary: List[torch.Tensor],
    affinity: Dict[Tuple[int, int], torch.Tensor],
    pair_weights: np.ndarray,
    pair_lambda: float = 1.30,
    max_iters: int = 30,
) -> Tuple[List[int], float, int]:
    selection = list(init_selection)
    n_images = len(selection)
    nodes_evaluated = 0

    for _ in range(max_iters):
        changed = False

        for i in range(n_images):
            candidate_scores = unary[i].clone()

            for j in range(n_images):
                if i == j:
                    continue
                w = float(pair_weights[i, j])
                candidate_scores += pair_lambda * w * affinity[(i, j)][:, selection[j]]

            nodes_evaluated += int(candidate_scores.numel())
            best_idx = int(torch.argmax(candidate_scores).item())

            if best_idx != selection[i]:
                selection[i] = best_idx
                changed = True

        if not changed:
            break

    obj = _objective(selection, unary, affinity, pair_weights, pair_lambda)
    return selection, obj, nodes_evaluated


def _generate_initializations(
    unary: List[torch.Tensor],
    affinity: Dict[Tuple[int, int], torch.Tensor],
    n_masks: List[int],
    max_anchor_seeds: int = 8,
) -> List[List[int]]:
    n_images = len(unary)
    inits = []

    # Init 1: best unary candidate in each image
    init_unary = [int(torch.argmax(u).item()) for u in unary]
    inits.append(init_unary)

    # Anchor image with fewest candidates
    anchor = int(np.argmin(n_masks))
    anchor_order = torch.argsort(unary[anchor], descending=True).tolist()
    anchor_order = anchor_order[: min(max_anchor_seeds, len(anchor_order))]

    for anchor_candidate in anchor_order:
        sel = [0] * n_images
        sel[anchor] = int(anchor_candidate)

        for j in range(n_images):
            if j == anchor:
                continue
            sel[j] = int(torch.argmax(affinity[(j, anchor)][:, anchor_candidate]).item())

        inits.append(sel)

    # Remove duplicates
    unique = []
    seen = set()
    for init in inits:
        key = tuple(init)
        if key not in seen:
            seen.add(key)
            unique.append(init)

    return unique


def _compute_winning_scores(
    selection: List[int],
    affinity: Dict[Tuple[int, int], torch.Tensor],
    pair_weights: np.ndarray,
) -> Tuple[List[float], float]:
    n_images = len(selection)

    winning_scores = []
    for i in range(n_images):
        weighted_sum = 0.0
        weight_total = 0.0
        for j in range(n_images):
            if i == j:
                continue
            w = float(pair_weights[i, j])
            weighted_sum += w * float(affinity[(i, j)][selection[i], selection[j]].item())
            weight_total += w
        score = weighted_sum / weight_total if weight_total > 0 else 0.0
        winning_scores.append(float(score))

    pair_sum = 0.0
    pair_wsum = 0.0
    for i in range(n_images):
        for j in range(i + 1, n_images):
            w = float(pair_weights[i, j])
            pair_sum += w * float(affinity[(i, j)][selection[i], selection[j]].item())
            pair_wsum += w

    global_affinity = pair_sum / pair_wsum if pair_wsum > 0 else float(np.mean(winning_scores))
    return winning_scores, float(global_affinity)


def solve_consensus_with_stats(
    embeddings_list: List[List[torch.Tensor]],
    masks_list: Optional[List[List[Dict]]] = None,
    image_shapes: Optional[List[Tuple[int, int]]] = None,
) -> Dict:
    """
    Mask-aware robust consensus solver.

    If masks_list/image_shapes are missing, falls back to a legacy embedding-only consensus.
    """
    n_images = len(embeddings_list)

    if masks_list is None or image_shapes is None:
        return _legacy_embedding_only_consensus(embeddings_list)

    if n_images == 0:
        return {
            "winning_indices": [],
            "winning_scores": [],
            "global_affinity": 0.0,
            "nodes_evaluated": 0,
            "solver_name": "mask_aware_consensus_v2",
            "objective_value": 0.0,
            "restarts": 0,
            "valid_candidates_per_image": [],
        }

    if n_images == 1:
        return {
            "winning_indices": [0],
            "winning_scores": [1.0],
            "global_affinity": 1.0,
            "nodes_evaluated": 0,
            "solver_name": "mask_aware_consensus_v2",
            "objective_value": 1.0,
            "restarts": 0,
            "valid_candidates_per_image": [1],
        }

    if len(masks_list) != n_images or len(image_shapes) != n_images:
        raise ValueError("embeddings_list, masks_list, and image_shapes must have the same length")

    # Truncate to aligned counts per image if ever needed
    aligned_embeddings = []
    aligned_masks = []
    original_index_maps = []

    for embs, masks in zip(embeddings_list, masks_list):
        n = min(len(embs), len(masks))
        if n == 0:
            return {
                "winning_indices": [0] * n_images,
                "winning_scores": [0.0] * n_images,
                "global_affinity": 0.0,
                "nodes_evaluated": 0,
                "solver_name": "mask_aware_consensus_v2",
                "objective_value": 0.0,
                "restarts": 0,
                "valid_candidates_per_image": [0] * n_images,
            }

        aligned_embeddings.append(embs[:n])
        aligned_masks.append(masks[:n])
        original_index_maps.append(list(range(n)))

    # Build candidate metadata and valid subsets
    reduced_embeddings = []
    reduced_metas = []
    reduced_to_original = []
    valid_counts = []

    for i in range(n_images):
        metas = [_extract_meta(m, image_shapes[i]) for m in aligned_masks[i]]

        valid_idx = [idx for idx, meta in enumerate(metas) if _candidate_is_valid(meta)]

        # If all candidates were filtered out, keep them all rather than crashing.
        if len(valid_idx) == 0:
            valid_idx = list(range(len(aligned_masks[i])))

        reduced_embeddings.append([aligned_embeddings[i][idx] for idx in valid_idx])
        reduced_metas.append([metas[idx] for idx in valid_idx])
        reduced_to_original.append([original_index_maps[i][idx] for idx in valid_idx])
        valid_counts.append(len(valid_idx))

    # Normalize embeddings
    normalized = [_normalize_embeddings(embs) for embs in reduced_embeddings]

    emb_dims = [x.shape[1] for x in normalized]
    if not all(d == emb_dims[0] for d in emb_dims):
        raise ValueError(f"Embedding dimensions mismatch across images: {emb_dims}")

    n_masks = [x.shape[0] for x in normalized]
    nodes_evaluated = 0

    # Build pairwise affinities
    affinity: Dict[Tuple[int, int], torch.Tensor] = {}
    pair_weights = np.zeros((n_images, n_images), dtype=np.float32)

    for i in range(n_images):
        for j in range(i + 1, n_images):
            Aij, wij, evals = _build_pair_affinity(
                normalized[i],
                normalized[j],
                reduced_metas[i],
                reduced_metas[j],
            )
            affinity[(i, j)] = Aij
            affinity[(j, i)] = Aij.T
            pair_weights[i, j] = wij
            pair_weights[j, i] = wij
            nodes_evaluated += evals

    unary = _compute_unary_scores(reduced_metas)
    initializations = _generate_initializations(unary, affinity, n_masks, max_anchor_seeds=8)

    best_selection = None
    best_objective = -1e18

    for init in initializations:
        selection, obj, evals = _coordinate_ascent(
            init_selection=init,
            unary=unary,
            affinity=affinity,
            pair_weights=pair_weights,
            pair_lambda=1.30,
            max_iters=30,
        )
        nodes_evaluated += evals

        if obj > best_objective:
            best_objective = obj
            best_selection = selection

    if best_selection is None:
        best_selection = [0] * n_images

    winning_scores, global_affinity = _compute_winning_scores(
        selection=best_selection,
        affinity=affinity,
        pair_weights=pair_weights,
    )

    # Map back to original candidate indices expected by app.py
    winning_indices = [
        int(reduced_to_original[i][best_selection[i]])
        for i in range(n_images)
    ]

    return {
        "winning_indices": winning_indices,
        "winning_scores": [float(x) for x in winning_scores],
        "global_affinity": float(global_affinity),
        "nodes_evaluated": int(nodes_evaluated),
        "solver_name": "mask_aware_consensus_v2",
        "objective_value": float(best_objective),
        "restarts": int(len(initializations)),
        "valid_candidates_per_image": [int(x) for x in valid_counts],
    }


def solve_consensus(
    embeddings_list: List[List[torch.Tensor]],
    masks_list: Optional[List[List[Dict]]] = None,
    image_shapes: Optional[List[Tuple[int, int]]] = None,
) -> List[int]:
    return solve_consensus_with_stats(
        embeddings_list=embeddings_list,
        masks_list=masks_list,
        image_shapes=image_shapes,
    )["winning_indices"]
