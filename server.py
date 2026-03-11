#!/usr/bin/env python3
"""
server.py — Servidor HTTP para o Sports Predictor (Futebol + NBA)
Serve a interface web (index.html) e processa pedidos de previsão.
Integra com a API pública da ESPN para dados reais de equipas.

Uso:
  python3 server.py          # porta 8000 (default)
  python3 server.py 9000     # porta personalizada

Endpoints:
  GET  /                                          → index.html
  POST /predict                                   → calcula probabilidades (futebol)
  POST /predict/nba                               → calcula probabilidades (NBA)
  GET  /api/leagues                               → ligas de futebol
  GET  /api/teams?league=por.1                    → equipas de uma liga
  GET  /api/stats?team_id=2250&league=por.1       → stats reais (futebol)
  GET  /api/h2h?home_id=X&away_id=Y&league=Z      → head-to-head (futebol)
  GET  /api/nba/teams                             → equipas NBA
  GET  /api/nba/stats?team_id=13                  → stats reais (NBA)
  GET  /api/nba/h2h?home_id=X&away_id=Y           → head-to-head (NBA)
"""

import sys
import json
import os
import urllib.request
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler

# Importar os motores do predictor.py
sys.path.insert(0, os.path.dirname(__file__))
from predictor import (
    apply_defaults, run_poisson, run_heuristic, format_output,
    apply_defaults_nba, run_nba, format_output_nba,
)

PORT          = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ESPN_BASE     = "https://site.api.espn.com/apis/site/v2/sports/soccer"
ESPN_NBA_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"

# ─────────────────────────────────────────────────────────────────────────────
# LIGAS SUPORTADAS
# ─────────────────────────────────────────────────────────────────────────────

LEAGUES = {
    "por.1":                  "🇵🇹 Liga Portugal",
    "eng.1":                  "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League",
    "esp.1":                  "🇪🇸 La Liga",
    "ger.1":                  "🇩🇪 Bundesliga",
    "ita.1":                  "🇮🇹 Serie A",
    "fra.1":                  "🇫🇷 Ligue 1",
    "uefa.champions":         "🌍 Champions League",
    "uefa.europa":            "🌍 Europa League",
    "ned.1":                  "🇳🇱 Eredivisie",
    "tur.1":                  "🇹🇷 Süper Lig",
}

# ─────────────────────────────────────────────────────────────────────────────
# ESPN API — helpers
# ─────────────────────────────────────────────────────────────────────────────

def fetch_json_url(url: str) -> dict:
    """Faz GET a um URL e devolve o JSON. Usa apenas stdlib."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; FootballPredictor/1.0)"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_stat(stats_list: list, name: str, default: float = 0.0) -> float:
    """Extrai o valor de uma stat pelo nome na lista de stats da ESPN."""
    for s in stats_list:
        if s.get("name") == name:
            return float(s.get("value", default))
    return default


def fetch_teams_list(league_id: str) -> list:
    """
    Devolve lista [{id, name}] das equipas de uma liga via ESPN API.
    URL: /teams
    """
    url = f"{ESPN_BASE}/{league_id}/teams"
    data = fetch_json_url(url)
    teams = []
    raw = (
        data.get("sports",  [{}])[0]
            .get("leagues", [{}])[0]
            .get("teams",   [])
    )
    for t in raw:
        team = t.get("team", {})
        tid  = team.get("id")
        name = team.get("displayName", "")
        if tid and name:
            teams.append({"id": str(tid), "name": name})
    return sorted(teams, key=lambda x: x["name"])


def calculate_form_from_events(events: list, team_id_str: str) -> dict:
    """
    Percorre os eventos do calendário de uma equipa e calcula
    o registo dos últimos 5 jogos concluídos.

    Devolve: {"wins", "draws", "losses", "gf", "ga", "recent_games"}
    Cada entrada em recent_games: {date, opponent, gf, ga, result, home_away}
    """
    completed = []
    for evt in events:
        competition = evt.get("competitions", [{}])[0]
        if not competition.get("status", {}).get("type", {}).get("completed"):
            continue
        comps = competition.get("competitors", [])
        if len(comps) < 2:
            continue

        our  = next((c for c in comps if str(c.get("id")) == team_id_str), None)
        opp  = next((c for c in comps if str(c.get("id")) != team_id_str), None)
        if not our or not opp:
            continue

        sc_our = our.get("score", {})
        sc_opp = opp.get("score", {})
        gf = float(sc_our.get("value", 0) if isinstance(sc_our, dict) else (sc_our or 0))
        ga = float(sc_opp.get("value", 0) if isinstance(sc_opp, dict) else (sc_opp or 0))
        completed.append({
            "gf":       gf,
            "ga":       ga,
            "date":     evt.get("date", ""),
            "opponent": opp.get("team", {}).get("displayName", ""),
            "home_away": our.get("homeAway", ""),
        })

    completed.sort(key=lambda x: x["date"], reverse=True)
    last5 = completed[:5]

    wins   = sum(1 for m in last5 if m["gf"] > m["ga"])
    draws  = sum(1 for m in last5 if m["gf"] == m["ga"])
    losses = sum(1 for m in last5 if m["gf"] < m["ga"])
    gf     = int(sum(m["gf"] for m in last5))
    ga     = int(sum(m["ga"] for m in last5))

    recent_games = [
        {
            "date":      m["date"][:10],
            "opponent":  m["opponent"],
            "gf":        int(m["gf"]),
            "ga":        int(m["ga"]),
            "result":    "W" if m["gf"] > m["ga"] else ("D" if m["gf"] == m["ga"] else "L"),
            "home_away": m["home_away"],
        }
        for m in last5
    ]

    return {
        "wins": wins, "draws": draws, "losses": losses, "gf": gf, "ga": ga,
        "recent_games": recent_games,
    }


def fetch_team_stats(team_id: str, league_id: str) -> dict:
    """
    Busca estatísticas reais de uma equipa via ESPN API.

    Calcula:
      avg_scored    = golos marcados / jogos disputados (temporada)
      avg_conceded  = golos sofridos / jogos disputados
      home_win_rate = vitórias em casa / jogos em casa
      away_win_rate = vitórias fora / jogos fora
      form          = resultado dos últimos 5 jogos
    """
    # ── 1. Record da temporada ───────────────────────────────────────────────
    url  = f"{ESPN_BASE}/{league_id}/teams/{team_id}"
    data = fetch_json_url(url)
    team_data = data.get("team", {})

    items = team_data.get("record", {}).get("items", [])
    if not items:
        # Ligas UEFA não têm record por equipa — usar liga doméstica como fallback
        dl_slug = (
            team_data.get("defaultLeague", {}).get("midsizeName", "") or ""
        ).lower()
        if not dl_slug:
            # Segunda tentativa: endpoint genérico de equipas de futebol
            try:
                generic = fetch_json_url(
                    f"https://site.api.espn.com/apis/site/v2/sports/soccer/teams/{team_id}"
                )
                dl_slug = (
                    generic.get("team", {})
                           .get("defaultLeague", {})
                           .get("midsizeName", "")
                    or ""
                ).lower()
            except Exception:
                pass
        if dl_slug and dl_slug != league_id:
            return fetch_team_stats(team_id, dl_slug)
        raise ValueError(f"Sem dados de record para equipa {team_id}")
    stats = items[0].get("stats", [])

    games_played  = get_stat(stats, "gamesPlayed")  or 1
    goals_for     = get_stat(stats, "pointsFor")
    goals_against = get_stat(stats, "pointsAgainst")
    home_games    = get_stat(stats, "homeGamesPlayed") or 1
    home_wins     = get_stat(stats, "homeWins")
    away_games    = get_stat(stats, "awayGamesPlayed") or 1
    away_wins     = get_stat(stats, "awayWins")

    avg_scored    = goals_for   / games_played
    avg_conceded  = goals_against / games_played
    home_win_rate = home_wins   / home_games
    away_win_rate = away_wins   / away_games

    # ── 2. Forma recente (últimos 5 jogos) ──────────────────────────────────
    sched_url  = f"{ESPN_BASE}/{league_id}/teams/{team_id}/schedule"
    sched_data = fetch_json_url(sched_url)
    events     = sched_data.get("events", [])
    form       = calculate_form_from_events(events, str(team_id))

    return {
        "name":          team_data.get("displayName", ""),
        "avg_scored":    round(avg_scored,    3),
        "avg_conceded":  round(avg_conceded,  3),
        "home_win_rate": round(home_win_rate, 3),
        "away_win_rate": round(away_win_rate, 3),
        "form":          form,
        "injuries":      0,
        "_source":       "ESPN API",
        "_games_played": int(games_played),
        "_goals_for":    int(goals_for),
        "_goals_against":int(goals_against),
    }


def fetch_h2h(home_id: str, away_id: str, league_id: str) -> dict:
    """
    Extrai o historial direto entre duas equipas do calendário da equipa de casa.
    Devolve: {"home_wins", "away_wins", "draws", "matches"}
    """
    sched_url  = f"{ESPN_BASE}/{league_id}/teams/{home_id}/schedule"
    sched_data = fetch_json_url(sched_url)
    events     = sched_data.get("events", [])

    home_wins = away_wins = draws = 0
    matches = []

    for evt in events:
        competition = evt.get("competitions", [{}])[0]
        if not competition.get("status", {}).get("type", {}).get("completed"):
            continue
        comps = competition.get("competitors", [])
        if len(comps) < 2:
            continue
        ids = {str(c.get("id")) for c in comps}
        if str(home_id) not in ids or str(away_id) not in ids:
            continue

        hc = next(c for c in comps if str(c.get("id")) == str(home_id))
        ac = next(c for c in comps if str(c.get("id")) == str(away_id))
        gf = float(hc.get("score", {}).get("value", 0))
        ga = float(ac.get("score", {}).get("value", 0))

        if gf > ga:
            home_wins += 1
        elif gf < ga:
            away_wins += 1
        else:
            draws += 1

        matches.append({
            "date": evt.get("date", "")[:10],
            "home_goals": int(gf),
            "away_goals": int(ga),
        })

    return {
        "home_wins": home_wins,
        "away_wins": away_wins,
        "draws":     draws,
        "matches":   matches,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NBA API — funções de fetch
# ─────────────────────────────────────────────────────────────────────────────

def fetch_nba_teams() -> list:
    """
    Devolve lista [{id, name}] das 30 equipas NBA via ESPN API.
    Path: sports[0].leagues[0].teams[n].team.{id, displayName}
    """
    data = fetch_json_url(f"{ESPN_NBA_BASE}/teams")
    teams = []
    raw = (
        data.get("sports",  [{}])[0]
            .get("leagues", [{}])[0]
            .get("teams",   [])
    )
    for t in raw:
        team = t.get("team", {})
        tid  = team.get("id")
        name = team.get("displayName", "")
        if tid and name:
            teams.append({"id": str(tid), "name": name})
    return sorted(teams, key=lambda x: x["name"])


def _nba_get_stat(items: list, item_type: str, stat_name: str, default: float = 0.0) -> float:
    """
    Extrai um valor de stat da lista record.items filtrada por type.
    item_type: "total" | "home" | "road"
    """
    item = next((i for i in items if i.get("type") == item_type), None)
    if not item:
        return default
    for s in item.get("stats", []):
        if s.get("name") == stat_name:
            return float(s.get("value", default))
    return default


def fetch_nba_team_stats(team_id: str) -> dict:
    """
    Busca estatísticas reais de uma equipa NBA via ESPN API.

    Estrutura ESPN NBA:
      record.items[type="total"].stats → avgPointsFor, avgPointsAgainst,
                                          wins, losses, gamesPlayed, winPercent
      record.items[type="home"].stats  → wins, losses, winPercent
      record.items[type="road"].stats  → wins, losses, winPercent

    Calcula:
      avg_scored    = avgPointsFor  (pontos por jogo marcados)
      avg_conceded  = avgPointsAgainst (pontos por jogo sofridos)
      home_win_rate = home_wins / home_games
      away_win_rate = away_wins / away_games
      form          = últimos 5 jogos do calendário
    """
    url       = f"{ESPN_NBA_BASE}/teams/{team_id}"
    data      = fetch_json_url(url)
    team_data = data.get("team", {})
    items     = team_data.get("record", {}).get("items", [])

    if not items:
        raise ValueError(f"Sem dados de record para equipa NBA {team_id}")

    # Médias de pontos (já calculadas pela ESPN)
    avg_scored   = _nba_get_stat(items, "total", "avgPointsFor")
    avg_conceded = _nba_get_stat(items, "total", "avgPointsAgainst")
    games_played = _nba_get_stat(items, "total", "gamesPlayed") or 1
    total_wins   = _nba_get_stat(items, "total", "wins")
    total_losses = _nba_get_stat(items, "total", "losses")

    # Home/Away win rate — wins e losses em items separados
    home_wins    = _nba_get_stat(items, "home", "wins")
    home_losses  = _nba_get_stat(items, "home", "losses")
    away_wins    = _nba_get_stat(items, "road", "wins")
    away_losses  = _nba_get_stat(items, "road", "losses")
    home_games   = (home_wins + home_losses) or 1
    away_games   = (away_wins + away_losses) or 1

    home_win_rate = home_wins / home_games
    away_win_rate = away_wins / away_games

    # Forma recente (últimos 5 jogos) — reutiliza a função de futebol
    # Nota: na NBA competitors[n].id (não team.id) — mesma estrutura
    sched_url  = f"{ESPN_NBA_BASE}/teams/{team_id}/schedule"
    sched_data = fetch_json_url(sched_url)
    events     = sched_data.get("events", [])
    form       = calculate_form_from_events(events, str(team_id))

    return {
        "name":           team_data.get("displayName", ""),
        "avg_scored":     round(avg_scored,    1),
        "avg_conceded":   round(avg_conceded,  1),
        "home_win_rate":  round(home_win_rate, 3),
        "away_win_rate":  round(away_win_rate, 3),
        "form":           form,
        "injuries":       0,
        "_source":        "ESPN API",
        "_games_played":  int(games_played),
        "_total_wins":    int(total_wins),
        "_total_losses":  int(total_losses),
    }


def fetch_nba_h2h(home_id: str, away_id: str) -> dict:
    """
    Historial direto entre duas equipas NBA, do calendário de casa.
    """
    sched_url  = f"{ESPN_NBA_BASE}/teams/{home_id}/schedule"
    sched_data = fetch_json_url(sched_url)
    events     = sched_data.get("events", [])

    home_wins = away_wins = 0
    matches   = []

    for evt in events:
        competition = evt.get("competitions", [{}])[0]
        if not competition.get("status", {}).get("type", {}).get("completed"):
            continue
        comps = competition.get("competitors", [])
        if len(comps) < 2:
            continue
        ids = {str(c.get("id")) for c in comps}
        if str(home_id) not in ids or str(away_id) not in ids:
            continue

        hc = next(c for c in comps if str(c.get("id")) == str(home_id))
        ac = next(c for c in comps if str(c.get("id")) == str(away_id))
        gf = float(hc.get("score", {}).get("value", 0))
        ga = float(ac.get("score", {}).get("value", 0))

        (home_wins if gf > ga else away_wins).__class__  # contagem abaixo
        if gf > ga:
            home_wins += 1
        else:
            away_wins += 1

        matches.append({
            "date":       evt.get("date", "")[:10],
            "home_pts":   int(gf),
            "away_pts":   int(ga),
        })

    return {
        "home_wins": home_wins,
        "away_wins": away_wins,
        "draws":     0,       # NBA não tem empates
        "matches":   matches,
    }


def fetch_nba_recent_games() -> list:
    """
    Busca jogos recentes da NBA dos últimos 7 dias via ESPN scoreboard API.
    Devolve lista ordenada por data (mais recente primeiro).
    URL: /scoreboard?dates=YYYYMMDD
    """
    from datetime import datetime, timedelta, timezone

    def parse_score(val):
        if isinstance(val, dict):
            return int(float(val.get("value", 0)))
        try:
            return int(float(val))
        except (TypeError, ValueError):
            return None

    games = []
    seen  = set()
    today = datetime.now(timezone.utc)

    for i in range(7):
        date     = today - timedelta(days=i)
        date_str = date.strftime("%Y%m%d")
        try:
            url  = f"{ESPN_NBA_BASE}/scoreboard?dates={date_str}&limit=30"
            data = fetch_json_url(url)
            for event in data.get("events", []):
                eid = event.get("id")
                if eid in seen:
                    continue
                seen.add(eid)

                comp         = event.get("competitions", [{}])[0]
                competitors  = comp.get("competitors", [])
                status_type  = comp.get("status", {}).get("type", {})
                completed    = bool(status_type.get("completed"))
                short_detail = status_type.get("shortDetail", "")

                home = next((c for c in competitors if c.get("homeAway") == "home"), None)
                away = next((c for c in competitors if c.get("homeAway") == "away"), None)
                if not home or not away:
                    continue

                home_score = parse_score(home.get("score")) if completed else None
                away_score = parse_score(away.get("score")) if completed else None
                winner = None
                if completed and home_score is not None and away_score is not None:
                    winner = (
                        home.get("team", {}).get("displayName", "")
                        if home_score > away_score
                        else away.get("team", {}).get("displayName", "")
                    )

                games.append({
                    "id":         eid,
                    "date":       event.get("date", "")[:10],
                    "home_team":  home.get("team", {}).get("displayName", ""),
                    "home_id":    str(home.get("team", {}).get("id", "")),
                    "away_team":  away.get("team", {}).get("displayName", ""),
                    "away_id":    str(away.get("team", {}).get("id", "")),
                    "home_score": home_score,
                    "away_score": away_score,
                    "completed":  completed,
                    "status":     short_detail,
                    "winner":     winner,
                })
        except Exception:
            pass  # data sem jogos — ignorar

    games.sort(key=lambda x: x["date"], reverse=True)
    return games


# ─────────────────────────────────────────────────────────────────────────────
# HTTP HANDLER
# ─────────────────────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # silenciar logs de acesso

    # ── GET ──────────────────────────────────────────────────────────────────

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path   = parsed.path
        params = dict(urllib.parse.parse_qsl(parsed.query))

        if path in ("/", "/index.html"):
            self._serve_file("index.html", "text/html; charset=utf-8")

        elif path == "/api/leagues":
            data = [{"id": k, "name": v} for k, v in LEAGUES.items()]
            self._json(200, data)

        elif path == "/api/teams":
            league = params.get("league", "por.1")
            try:
                teams = fetch_teams_list(league)
                self._json(200, teams)
            except Exception as e:
                self._json(500, {"error": str(e)})

        elif path == "/api/stats":
            team_id = params.get("team_id")
            league  = params.get("league", "por.1")
            if not team_id:
                self._json(400, {"error": "Parâmetro 'team_id' obrigatório."})
                return
            try:
                stats = fetch_team_stats(team_id, league)
                self._json(200, stats)
            except Exception as e:
                self._json(500, {"error": str(e)})

        elif path == "/api/h2h":
            home_id = params.get("home_id")
            away_id = params.get("away_id")
            league  = params.get("league", "por.1")
            if not home_id or not away_id:
                self._json(400, {"error": "Parâmetros 'home_id' e 'away_id' obrigatórios."})
                return
            try:
                h2h = fetch_h2h(home_id, away_id, league)
                self._json(200, h2h)
            except Exception as e:
                self._json(500, {"error": str(e)})

        # ── NBA endpoints ──────────────────────────────────────────────────────

        elif path == "/api/nba/teams":
            try:
                self._json(200, fetch_nba_teams())
            except Exception as e:
                self._json(500, {"error": str(e)})

        elif path == "/api/nba/stats":
            team_id = params.get("team_id")
            if not team_id:
                self._json(400, {"error": "Parâmetro 'team_id' obrigatório."})
                return
            try:
                self._json(200, fetch_nba_team_stats(team_id))
            except Exception as e:
                self._json(500, {"error": str(e)})

        elif path == "/api/nba/h2h":
            home_id = params.get("home_id")
            away_id = params.get("away_id")
            if not home_id or not away_id:
                self._json(400, {"error": "Parâmetros 'home_id' e 'away_id' obrigatórios."})
                return
            try:
                self._json(200, fetch_nba_h2h(home_id, away_id))
            except Exception as e:
                self._json(500, {"error": str(e)})

        elif path == "/api/nba/recent-games":
            try:
                self._json(200, fetch_nba_recent_games())
            except Exception as e:
                self._json(500, {"error": str(e)})

        else:
            self._json(404, {"error": "Not found"})

    # ── POST ──────────────────────────────────────────────────────────────────

    def do_POST(self):
        parsed_path = urllib.parse.urlparse(self.path).path
        if parsed_path not in ("/predict", "/predict/nba"):
            self._json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as e:
            self._json(400, {"error": f"JSON inválido: {e}"})
            return

        show_debug = payload.pop("_debug", False)

        # ── NBA ──────────────────────────────────────────────────────────────
        if parsed_path == "/predict/nba":
            try:
                data  = apply_defaults_nba(payload)
                probs, debug_info = run_nba(data)
                output = format_output_nba(data, probs, debug_info, show_debug)
            except Exception as e:
                self._json(500, {"error": str(e)})
                return
            self._json(200, output)
            return

        # ── Futebol ──────────────────────────────────────────────────────────
        model_name       = payload.pop("_model",    "poisson")
        include_optional = payload.pop("_optional", True)
        if model_name not in ("poisson", "heuristic"):
            model_name = "poisson"
        try:
            data   = apply_defaults(payload)
            run_fn = run_poisson if model_name == "poisson" else run_heuristic
            probs, debug_info = run_fn(data, include_optional=include_optional)
            output = format_output(data, probs, model_name, debug_info, show_debug)
        except Exception as e:
            self._json(500, {"error": str(e)})
            return
        self._json(200, output)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _json(self, status: int, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type",   "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, filename: str, content_type: str):
        filepath = os.path.join(BASE_DIR, filename)
        try:
            with open(filepath, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type",   content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self._json(404, {"error": "file not found"})


if __name__ == "__main__":
    # Em cloud (Railway, Render, etc.) usa 0.0.0.0 e a variável PORT do ambiente
    import os as _os
    PORT = int(_os.environ.get("PORT", PORT))
    HOST = "0.0.0.0"  # aceita ligações externas na cloud; localhost em local
    server = HTTPServer((HOST, PORT), Handler)
    print(f"Football Predictor → http://{HOST}:{PORT}", flush=True)
    server.serve_forever()
