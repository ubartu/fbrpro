"""Describe shirt crops and group the best-supported appearances within a shot."""
from collections import Counter


KIT_RADIUS = 14
KIT_SEPARATION = 25
MIN_SAMPLES = 3
MIN_SPAN_MS = 400
MIN_PLAYERS = 3


def shirt_appearance(cv2, np, frame, box):
    x1, y1, x2, y2 = map(round, box)
    width, height = x2-x1, y2-y1
    if width < 8 or height < 22:
        return None
    h, w = frame.shape[:2]
    left, right = max(0, x1-width//3), min(w, x2+width//3)
    top, bottom = max(0, y1+round(height*.22)), min(h, y1+round(height*.50))
    if bottom-top < 5 or right-left < 5:
        return None
    inset = max(1, width//5)
    start, stop = max(0, x1+inset-left), min(right-left, x2-inset-left)
    if stop-start < 4:
        return None
    patch = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2LAB).astype(np.float32)
    patch[:, :, 0] *= .55
    torso = patch[:, start:stop].reshape(-1, 3)
    outside = np.concatenate((patch[:, :max(0, x1-left)], patch[:, max(0, x2-left):]), axis=1).reshape(-1, 3)
    turf = outside[(outside[:, 1] < 122) & (outside[:, 2] > 130)]
    if len(turf) >= 8:
        grass = np.linalg.norm(torso-np.median(turf, axis=0), axis=1) < 17
        if (~grass).sum() < max(16, len(torso)*.40):
            return None
        torso = torso[~grass]
    # Keep the distribution so a striped kit is not reduced to one hue.
    return np.quantile(torso, (.25, .50, .75), axis=0).ravel().astype(np.float32)


def repeated(times):
    return len(times) >= MIN_SAMPLES and max(times)-min(times) >= MIN_SPAN_MS


def distance(np, appearances, center):
    differences = np.subtract(appearances, center, dtype=np.float32)
    differences = differences.reshape((*differences.shape[:-1], 3, 3))
    # Shade changes lightness more than chroma. Preserve the difference between
    # similarly dark kits, including keepers whose shirts would otherwise match.
    differences[..., 1:] *= 1.45
    return np.linalg.norm(differences, axis=(-2, -1)) / 3


def team_groups(samples, reference):
    """Return two co-visible groups, or abstain if the shirt evidence is weak."""
    import numpy as np

    profiles = {}
    for track, rows in samples.items():
        if not repeated(rows):
            continue
        values = np.asarray(list(rows.values()))
        center = np.median(values, axis=0)
        inliers = distance(np, values, center) <= KIT_RADIUS
        times = {ms for ms, keep in zip(rows, inliers) if keep}
        if repeated(times) and len(times) >= .65*len(rows):
            profiles[track] = (np.median(values[inliers], axis=0), times)
    if not profiles:
        return {}, reference, "two_co_visible_kits_unconfirmed"
    tracks = list(profiles)
    vectors = np.stack([profiles[t][0] for t in tracks])
    if reference is not None:
        assignments = assign_groups(np, tracks, vectors, profiles, reference)
        return assignments, reference, "inferred" if assignments else "kit_appearance_unmatched"
    if len(profiles) < 2*MIN_PLAYERS:
        return {}, reference, "two_co_visible_kits_unconfirmed"
    weights = np.asarray([min(30, len(profiles[t][1])) for t in tracks])
    # Bound prototype search on long uninterrupted recordings.
    seeds = np.argsort(-weights, kind="stable")[:80]
    neighbors = distance(np, vectors[None, :], vectors[seeds, None]) <= KIT_RADIUS
    support = neighbors @ weights
    modes = []
    for index in np.argsort(-support, kind="stable"):
        center = np.median(vectors[neighbors[index]], axis=0)
        if any(distance(np, center, other[0]) < KIT_SEPARATION for other in modes):
            continue
        nearby = np.flatnonzero(distance(np, vectors, center) <= KIT_RADIUS)
        counts = Counter(ms for i in nearby for ms in profiles[tracks[i]][1])
        co_visible = {ms for ms, n in counts.items() if n >= MIN_PLAYERS}
        if repeated(co_visible):
            modes.append((center, co_visible))
    if len(modes) > 2:
        return {}, reference, "multiple_recurring_kits"
    if len(modes) < 2 or not repeated(modes[0][1] & modes[1][1]):
        return {}, reference, "two_co_visible_kits_unconfirmed"
    centers = np.stack([mode[0] for mode in modes])
    centers = centers[np.argsort(centers[:, 3], kind="stable")]
    return assign_groups(np, tracks, vectors, profiles, centers), centers, "inferred"


def assign_groups(np, tracks, vectors, profiles, centers):
    distances = distance(np, vectors[:, None], centers)
    assignments = {}
    for track, scores in zip(tracks, distances):
        best = int(np.argmin(scores))
        if scores[best] <= KIT_RADIUS and scores[1-best]-scores[best] >= 6:
            assignments[track] = ("team_a" if best == 0 else "team_b", profiles[track][1])
    return assignments
