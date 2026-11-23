"""Pair observed closing runs with carriers and follow the resulting possession."""
from collections import defaultdict, deque
from math import dist


def estimate_presses(pairs, prepared, tracks, dead):
    frames = {frame["t_ms"]: {p["id"]: p for p in frame["players"]} for frame in prepared["frames"]}
    histories = defaultdict(deque)
    active = {}
    presses = []
    window_ms = prepared["context"].get("press_outcome_window_ms", 5000)
    def stopped(first, second):
        return any(stop.get("period", 1) == first.period and stop["start_ms"] < second.t_ms
                   and first.t_ms < stop["end_ms"] for stop in dead)

    for index, (first, second, state, carrier, team, shot) in enumerate(pairs):
        ongoing = set()
        if state == "player_control" and team and not stopped(first, second):
            a = frames.get(first.t_ms, {})
            b = frames.get(second.t_ms, {})
            target = b.get(carrier)
            if target and target["team"] == team and carrier in a:
                for track in (a.keys() & b.keys()) - {carrier}:
                    prior, defender = a[track], b[track]
                    side = defender["team"]
                    if (not side or side != prior["team"] or side == team
                            or side.startswith("kit:") != team.startswith("kit:")):
                        continue
                    start_pos, end_pos = (prior["x"], prior["y"]), (defender["x"], defender["y"])
                    target_pos = (target["x"], target["y"])
                    gap = dist(end_pos, target_pos)
                    if gap > 15 or dist(start_pos, end_pos) > 15*(second.t_ms-first.t_ms)/1000+1:
                        continue
                    key = (shot, carrier, team, track, side)
                    ongoing.add(key)
                    if key in active:
                        if gap+defender["error_m"]+target["error_m"] <= 6:
                            active[key].update(end_ms=second.t_ms, last_index=index)
                        else:
                            ongoing.remove(key)
                        continue
                    history = histories[key]
                    if not history:
                        old_target = a[carrier]
                        history.append((first.t_ms, start_pos, (old_target["x"], old_target["y"]),
                                        prior["error_m"]+old_target["error_m"]))
                    while history and second.t_ms-history[0][0] > 3000:
                        history.popleft()
                    error = defender["error_m"]+target["error_m"]
                    for stamp, origin, carrier_origin, uncertainty in history:
                        if (second.t_ms-stamp >= 500 and gap+error <= 5
                                and dist(origin, carrier_origin)-gap >= max(4, uncertainty+error)
                                and dist(origin, carrier_origin)-dist(end_pos, carrier_origin) >= max(2, uncertainty)):
                            episode = {"id": f"tracking:{second.period}:{second.t_ms}:press:{track}", "type": "press",
                                       "label": "Estimated press", "source": "tracking", "status": "estimated",
                                       "period": second.period, "t_ms": second.t_ms, "start_ms": stamp,
                                       "end_ms": second.t_ms, "track_id": track, "player_id": tracks[track].get("player_id"),
                                       "team_id": side, "carrier_track_id": carrier, "carrier_player_id": tracks[carrier].get("player_id"),
                                       "carrier_team_id": team, "actor_reason": "Closed down an observed opposing ball carrier",
                                       "last_index": index, "shot": shot}
                            active[key] = episode
                            break
                    history.append((second.t_ms, end_pos, target_pos, error))
        for key in set(active)-ongoing:
            presses.append(active.pop(key))
        for key in set(histories)-ongoing:
            del histories[key]
    presses.extend(active.values())

    for row in presses:
        start_index = row.pop("last_index")+1
        shot = row.pop("shot")
        deadline = row["end_ms"]+window_ms
        row.update(outcome="unknown", outcome_window_ms=window_ms,
                   outcome_reason="Continuation lost or footage ended before the outcome window")
        for index in range(start_index, len(pairs)):
            first, second, state, owner, team, subsequent_shot = pairs[index]
            if (first is None or second is None or state == "unobservable" or subsequent_shot != shot
                    or first.period != row["period"] or stopped(first, second)):
                break
            if first.t_ms < deadline and state == "player_control" and team == row["team_id"]:
                row.update(outcome="successful", outcome_reason="Pressing team gained observable control in the outcome window",
                           regain_ms=first.t_ms, next_control_track_id=owner,
                           next_control_player_id=tracks[owner].get("player_id"), next_control_team_id=team)
                break
            if state == "player_control" and team != row["carrier_team_id"]:
                break
            if second.t_ms >= deadline:
                row.update(outcome="unsuccessful", outcome_reason="No pressing-team regain in the observable outcome window")
                break
    return presses
