"""Reconnect compatible player fragments within one continuous camera shot."""
from collections import defaultdict
from dataclasses import dataclass
from math import dist, hypot, log

from .video_kits import distance


@dataclass
class Fragment:
    id: str
    first: int
    last: int
    head: list
    tail: list


def _anchor(row):
    l, t, r, b = row["bbox_xyxy"]
    return ((l+r)/2, b), b-t


def _at_edge(row, size):
    l, t, r, b = row["bbox_xyxy"]
    return l <= 5 or t <= 5 or r >= size[0]-6 or b >= size[1]-6


def _crowded(row, others, motion=None):
    l, t, r, b = row["bbox_xyxy"]
    if motion is not None:
        points = [motion @ [x, y, 1.] for x in (l, r) for y in (t, b)]
        xs, ys = zip(*points)
        l, t, r, b = min(xs), min(ys), max(xs), max(ys)
    for other in others:
        if other["track_id"] == row["track_id"]:
            continue
        x, y, right, bottom = other["bbox_xyxy"]
        if min(r, right)-max(l, x) > 0 and min(b, bottom)-max(t, y) > .25*min(b-t, bottom-y):
            return True
    return False


def _median_kit(np, rows, people):
    crops = [row["_shirt_appearance"] for index, row in rows
             if "_shirt_appearance" in row and not _crowded(row, people[index])]
    return np.median(crops, axis=0) if len(crops) >= 2 else None


def _jersey_number(rows):
    numbers = [row["jersey_number"] for _, row in rows if row.get("jersey_number") is not None]
    return numbers[0] if len(numbers) >= 2 and all(n == numbers[0] for n in numbers) else None


def _transport(np, motion, start, end):
    matrix = np.eye(3)
    for step in motion[start+1:end+1]:
        if step is None:
            return None
        matrix = np.vstack((step, [0, 0, 1])) @ matrix
    return matrix


def _link_cost(np, a, b, times, motion, kits, teams, numbers):
    if teams.get(a.id) and teams.get(b.id) and teams[a.id] != teams[b.id]:
        return None
    first_kit, next_kit = kits[a.id][1], kits[b.id][0]
    if first_kit is None or next_kit is None:
        return None
    appearance = float(distance(np, first_kit, next_kit))
    if appearance > 11:
        return None
    old_number, new_number = numbers[a.id][1], numbers[b.id][0]
    known_numbers = old_number is not None and new_number is not None
    if known_numbers and old_number != new_number:
        return None
    previous_i, previous = a.tail[-2]
    last_i, last = a.tail[-1]
    start_i, start = b.head[0]
    next_i, following = b.head[1]
    if not (0 < times[last_i]-times[previous_i] <= 400 and 0 < times[next_i]-times[start_i] <= 400):
        return None
    previous_motion = _transport(np, motion, previous_i, last_i)
    gap_motion = _transport(np, motion, last_i, start_i)
    following_motion = _transport(np, motion, start_i, next_i)
    if previous_motion is None or gap_motion is None or following_motion is None:
        return None
    p, _ = _anchor(previous)
    q, height = _anchor(last)
    r, next_height = _anchor(start)
    s, _ = _anchor(following)
    scale = hypot(*gap_motion[:2, 0])
    ratio = next_height/max(1., height*scale)
    if not .70 <= ratio <= 1.43:
        return None
    old_step = np.asarray(q)-(previous_motion @ [*p, 1.])[:2]
    new_step = (np.linalg.inv(following_motion) @ [*s, 1.])[:2]-r
    elapsed = times[start_i]-times[last_i]
    old_travel = (gap_motion[:2, :2] @ old_step)*elapsed/(times[last_i]-times[previous_i])
    new_travel = new_step*elapsed/(times[next_i]-times[start_i])
    moved = (gap_motion @ [*q, 1.])[:2]
    forward = dist(moved+old_travel, r)/next_height
    backward = dist(moved, r-new_travel)/next_height
    if min(forward, backward) > .55 or max(forward, backward) > 1.05:
        return None
    return (forward+backward)/2 + appearance/30 + .30*abs(log(ratio)) - (.25 if known_numbers else 0)


def reconnect_shot(np, frames, motion, size, teams=None):
    """Use clear crops and bidirectional motion before changing any track IDs."""
    if len(frames) < 5:
        return []
    teams = teams or {}
    times = [f["video_time_ms"] for f in frames]
    people = [[row for row in f["objects"] if row["observed"] and row["object_type"] == "player"] for f in frames]
    observed = defaultdict(list)
    for i, rows in enumerate(people):
        for row in rows:
            observed[row["track_id"]].append((i, row))
    fragments = [Fragment(id_, rows[0][0], rows[-1][0], rows[:5], rows[-5:])
                 for id_, rows in observed.items() if len(rows) >= 3]
    ends = defaultdict(list)
    kits = {f.id: (_median_kit(np, f.head, people), _median_kit(np, f.tail, people)) for f in fragments}
    numbers = {f.id: (_jersey_number(f.head), _jersey_number(f.tail)) for f in fragments}
    for fragment in fragments:
        row = fragment.tail[-1][1]
        after = fragment.last+1
        overlap = _crowded(row, people[fragment.last]) or (
            after < len(people) and motion[after] is not None and _crowded(row, people[after], motion[after]))
        if not _at_edge(row, size) and overlap:
            ends[fragment.last].append(fragment)

    options = []
    for b in fragments:
        if _at_edge(b.head[0][1], size):
            continue
        for i in range(b.first-2, -1, -1):
            if times[b.first]-times[i] > 1600:
                break
            for a in ends[i]:
                cost = _link_cost(np, a, b, times, motion, kits, teams, numbers)
                if cost is not None:
                    options.append((cost, a, b))

    by_old, by_new = defaultdict(list), defaultdict(list)
    for option in options:
        by_old[option[1].id].append(option)
        by_new[option[2].id].append(option)
    links = {}
    for cost, a, b in options:
        if any(candidate[0] <= cost+.22 and (candidate[1].id, candidate[2].id) != (a.id, b.id)
               for candidate in by_old[a.id]+by_new[b.id]):
            continue
        links[b.id] = a.id
    if not links:
        return []

    aliases = {}
    for track, earlier in links.items():
        while earlier in links:
            earlier = links[earlier]
        aliases[track] = earlier
    affected = set(aliases) | set(aliases.values())
    for frame in frames:
        present = {aliases.get(row["track_id"], row["track_id"]) for row in frame["objects"] if row["observed"]}
        rows = []
        for row in frame["objects"]:
            track = row["track_id"]
            target = aliases.get(track, track)
            if not row["observed"] and track in affected and (track != target or target in present):
                continue
            row["track_id"] = target
            rows.append(row)
        frame["objects"] = rows
    return [{"from_track_id": track, "to_track_id": target} for track, target in aliases.items()]
