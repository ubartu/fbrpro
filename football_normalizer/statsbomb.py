"""Adapter for StatsBomb Open Data JSON (events, lineups, optional 360)."""
import json

from .core import boolean, get_path, integer, number, point, read_records, time_ms


def slug(name):
    return str(name).lower().replace(" ", "_").replace("-", "_")


def normalize(args, bundle):
    identity = bundle.identity
    mid = identity.resolve("match", args.match_id)
    match_source = bundle.archive(args.match) if args.match else None
    match = None
    if args.match:
        content = json.loads(args.match.read_text(encoding="utf-8"))
        matches = content if isinstance(content, list) else [content]
        match = next((m for m in matches if str(m["match_id"]) == str(args.match_id)), None)
        if match is None:
            raise ValueError("requested match_id is not in match metadata file")
    lineups_source = bundle.archive(args.lineups)
    lineups = list(read_records(args.lineups))
    if len(lineups) != 2:
        raise ValueError("StatsBomb lineups must contain exactly two teams")
    for team in lineups:
        bundle.emit("teams", {"team_id": identity.resolve("team", team["team_id"]), "name": team["team_name"]}, team["team_id"], lineups_source)
    home = get_path(match, "home_team.home_team_id") if match else args.home_team_id
    away = get_path(match, "away_team.away_team_id") if match else args.away_team_id
    if home is None or away is None:
        raise ValueError("supply match metadata or both --home-team-id and --away-team-id")
    if {str(home), str(away)} != {str(t["team_id"]) for t in lineups}:
        raise ValueError("home/away team IDs do not agree with lineups")
    if match:
        comp, season = match.get("competition"), match.get("season")
        if comp:
            bundle.emit("competitions", {"competition_id": identity.resolve("competition", comp["competition_id"]),
                                        "name": comp["competition_name"], "country": comp.get("country_name")}, comp["competition_id"], match_source)
        if season and comp:
            bundle.emit("seasons", {"season_id": identity.resolve("season", season["season_id"], str(comp["competition_id"])),
                                   "competition_id": identity.resolve("competition", comp["competition_id"]), "name": season["season_name"]}, season["season_id"], match_source)
    length, width = args.pitch_length, args.pitch_width
    if (length is None) != (width is None):
        raise ValueError("supply both pitch dimensions or neither")
    # Open event coordinates represent a standardised 120x80 pitch, not surveyed metres.
    # A declared reference pitch is used if actual dimensions are unavailable.
    known = length is not None
    length, width = (length, width) if known else (105.0, 68.0)
    if length <= 0 or width <= 0:
        raise ValueError("pitch dimensions must be positive")
    if not known:
        bundle.warnings.append("Actual pitch dimensions unavailable; event/snapshot metres use a declared 105x68 reference pitch.")
    coord = {"system": "scaled", "x_max": 120, "y_max": 80}
    bundle.emit("matches", {"match_id": mid, "home_team_id": identity.resolve("team", home),
                            "away_team_id": identity.resolve("team", away),
                            "competition_id": identity.resolve("competition", get_path(match, "competition.competition_id")),
                            "season_id": identity.resolve("season", get_path(match, "season.season_id"), str(get_path(match, "competition.competition_id"))),
                            "date": match.get("match_date") if match else None,
                            "pitch_length_m": length if known else None, "pitch_width_m": width if known else None,
                            "coordinate_length_m": length, "coordinate_width_m": width,
                            "coordinate_frame": "team_attacking_left_to_right", "dimensions_assumed": not known,
                            "attacking_directions": None}, args.match_id, match_source)
    players_seen = set()
    for team in lineups:
        tid = identity.resolve("team", team["team_id"])
        for p in team["lineup"]:
            pid = identity.resolve("player", p["player_id"])
            if pid not in players_seen:
                bundle.emit("players", {"player_id": pid, "name": p["player_name"],
                                        "nickname": p.get("player_nickname"), "country": get_path(p, "country.name")}, p["player_id"], lineups_source)
                players_seen.add(pid)
            positions = []
            for pos in p.get("positions", []):
                positions.append({"position": pos.get("position"), "position_id": pos.get("position_id"),
                                  "from": pos.get("from"), "to": pos.get("to"),
                                  "from_period": pos.get("from_period"), "to_period": pos.get("to_period"),
                                  "start_reason": pos.get("start_reason"), "end_reason": pos.get("end_reason"),
                                  "clock_basis": "provider_match_clock"})
            bundle.emit("appearances", {"appearance_id": identity.resolve("appearance", f"{team['team_id']}:{p['player_id']}", args.match_id),
                                        "match_id": mid, "team_id": tid, "player_id": pid,
                                        "shirt_number": p.get("jersey_number"), "position_intervals": positions,
                                        "starter": any(x.get("start_reason") == "Starting XI" for x in positions) if "positions" in p else None,
                                        "minutes_played": None, "participated": bool(positions) if "positions" in p else None}, p["player_id"], lineups_source)
    event_source = bundle.archive(args.events)
    for i, event in enumerate(read_records(args.events), 1):
        kind = event["type"]["name"]
        payload = event.get(slug(kind), {})
        x, y = point(*(event.get("location") or [None, None])[:2], coord, length, width)
        end = payload.get("end_location") or [None, None]
        ex, ey = point(*end[:2], coord, length, width)
        outcome = get_path(payload, "outcome.name")
        # In StatsBomb, absence of pass outcome specifically denotes a completed pass.
        if kind == "Pass" and outcome is None:
            outcome = "complete"
        elif outcome is not None:
            outcome = slug(outcome)
        row = {"event_id": identity.resolve("event", event["id"], args.match_id), "match_id": mid,
               "period": integer(event["period"]), "period_time_ms": time_ms(event["timestamp"], "timestamp"),
               "team_id": identity.resolve("team", get_path(event, "team.id")),
               "player_id": identity.resolve("player", get_path(event, "player.id")),
               "type": slug(kind), "outcome": outcome,
               "x_m": x, "y_m": y, "end_x_m": ex, "end_y_m": ey,
               "coordinate_frame": "team_attacking_left_to_right", "coordinate_length_m": length, "coordinate_width_m": width,
               "recipient_id": identity.resolve("player", get_path(payload, "recipient.id")),
               "replacement_id": identity.resolve("player", get_path(payload, "replacement.id")),
               "under_pressure": boolean(event.get("under_pressure")),
               "duration_ms": None if event.get("duration") is None else number(event["duration"]) * 1000,
               "provider_minute": event.get("minute"), "provider_second": event.get("second"),
               "provider_index": event.get("index"), "provider_position": get_path(event, "position.name"),
               "possession_id": event.get("possession"), "possession_team_id": identity.resolve("team", get_path(event, "possession_team.id")),
               "play_pattern": get_path(event, "play_pattern.name"),
               "provider_attributes": payload, "related_event_ids": [identity.resolve("event", e, args.match_id) for e in event.get("related_events", [])]}
        bundle.emit("events", row, event["id"], event_source, i)
    if args.snapshots:
        snapshot_source = bundle.archive(args.snapshots)
        for i, snapshot in enumerate(read_records(args.snapshots), 1):
            players = []
            for p in snapshot.get("freeze_frame", []):
                x, y = point(*p["location"][:2], coord, length, width)
                players.append({"player_id": identity.resolve("player", get_path(p, "player.id")),
                                "x_m": x, "y_m": y, "teammate": boolean(p.get("teammate")),
                                "actor": boolean(p.get("actor")), "keeper": boolean(p.get("keeper"))})
            area = snapshot.get("visible_area") or []
            if len(area) % 2:
                raise ValueError("360 visible_area has an odd number of coordinates")
            polygon = [list(point(area[j], area[j + 1], coord, length, width)) for j in range(0, len(area), 2)]
            bundle.emit("snapshots", {"snapshot_id": identity.resolve("snapshot", snapshot["event_uuid"], args.match_id),
                                      "event_id": identity.resolve("event", snapshot["event_uuid"], args.match_id),
                                      "match_id": mid, "players": players, "visible_area": polygon,
                                      "coordinate_frame": "team_attacking_left_to_right",
                                      "coordinate_length_m": length, "coordinate_width_m": width}, snapshot["event_uuid"], snapshot_source, i)
        bundle.warnings.append("360 snapshots are event-time observations, not continuous tracking. Anonymous snapshot players remain unidentified.")
