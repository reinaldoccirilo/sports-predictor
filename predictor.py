#!/usr/bin/env python3
"""
predictor.py — Preditor de Probabilidades de Futebol
=====================================================
Gera probabilidades 1X2, Over/Under 2.5 e BTTS.
SEM conselhos de aposta, SEM odds — apenas percentagens.

Motores disponíveis:
  poisson   — distribuição de Poisson via golos esperados (recomendado)
  heuristic — pesos heurísticos com softmax

Uso:
  python predictor.py --model poisson   --input game.json
  python predictor.py --model heuristic --demo
  python predictor.py --model poisson   --demo --debug
  python predictor.py --test
"""

import math
import json
import argparse
import sys
import logging
from typing import Optional, Tuple, Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES E DEFAULTS GLOBAIS
# ─────────────────────────────────────────────────────────────────────────────

# Média de golos por jogo da liga (âncora de shrinkage)
# Liga europeia típica: ~2.7 total/jogo → ~1.35 por equipa
LEAGUE_AVG_GOALS: float = 1.35

# Multiplicador de vantagem em casa para lambda_home no modelo Poisson
# Representa aprox. 5–15% de boost nos golos esperados em casa
HOME_ADVANTAGE_DEFAULT: float = 1.10

# Peso de shrinkage (regressão à média da liga)
# 0.0 = sem shrinkage, 1.0 = tudo league avg
# Valor 0.2 → 80% estatísticas da equipa, 20% média da liga
SHRINKAGE_WEIGHT: float = 0.20

# Pesos do motor heurístico (devem somar 1.0)
HEURISTIC_WEIGHTS: Dict[str, float] = {
    "form":      0.35,   # forma recente (últimos 5 jogos)
    "goals":     0.35,   # rácio golos marcados/sofridos (temporada)
    "home_away": 0.20,   # taxa de vitória em casa ou fora
    "h2h":       0.10,   # historial de confrontos diretos
}

# Piso mínimo para probabilidade de empate (%)
# Garante que o empate nunca fica a 0% ou próximo de zero
MIN_DRAW_PCT: float = 15.0

# Número máximo de golos a calcular explicitamente na Poisson
# O bucket MAX_K absorve toda a probabilidade restante (P(X >= MAX_K))
MAX_K: int = 7

# Valores padrão de forma quando não fornecidos
DEFAULT_FORM: Dict[str, int] = {
    "wins": 2, "draws": 1, "losses": 2, "gf": 6, "ga": 6
}

# Médias padrão de golos quando não fornecidas
DEFAULT_AVG_SCORED: float = 1.35
DEFAULT_AVG_CONCEDED: float = 1.35

# Taxas de vitória padrão quando não fornecidas
DEFAULT_HOME_WIN_RATE: float = 0.50
DEFAULT_AWAY_WIN_RATE: float = 0.35


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

log = logging.getLogger("predictor")


def setup_logging(debug: bool) -> None:
    """Configura logging: DEBUG para stderr se --debug, senão silêncio."""
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
        stream=sys.stderr,
    )


# ─────────────────────────────────────────────────────────────────────────────
# VALIDAÇÃO E PREENCHIMENTO DE DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

def apply_defaults(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preenche campos em falta com valores padrão razoáveis.
    Emite log.warning para cada campo omitido para transparência.

    Esquema esperado (todos os campos são opcionais exceto home/away):
    {
      "home": {
        "name": str,
        "form": {"wins":int, "draws":int, "losses":int, "gf":int, "ga":int},
        "avg_scored": float,
        "avg_conceded": float,
        "home_win_rate": float,   # 0..1
        "injuries": int
      },
      "away": {
        "name": str,
        "form": {...},
        "avg_scored": float,
        "avg_conceded": float,
        "away_win_rate": float,   # 0..1
        "injuries": int
      },
      "h2h": {"home_wins":int, "away_wins":int, "draws":int},  # opcional
      "date": str,                # opcional
      "home_advantage": float,    # opcional, override do default
      "market_odds": {...}        # opcional, ignorado se não fornecido
    }
    """
    d: Dict[str, Any] = dict(raw)

    d.setdefault("date", None)
    d.setdefault("h2h", None)
    d.setdefault("market_odds", None)
    d.setdefault("home_advantage", HOME_ADVANTAGE_DEFAULT)

    for side, label, wr_key, default_wr in [
        ("home", "casa",  "home_win_rate", DEFAULT_HOME_WIN_RATE),
        ("away", "fora",  "away_win_rate", DEFAULT_AWAY_WIN_RATE),
    ]:
        if side not in d or not isinstance(d.get(side), dict):
            log.warning("Equipa '%s' não definida ou inválida → usando defaults.", label)
            d[side] = {}

        team = d[side]

        # Nome
        team.setdefault("name", f"Equipa {label.capitalize()}")

        # Forma recente
        if "form" not in team or not isinstance(team.get("form"), dict):
            log.warning("[%s] 'form' em falta → usando defaults %s", label, DEFAULT_FORM)
            team["form"] = dict(DEFAULT_FORM)
        else:
            for k, v in DEFAULT_FORM.items():
                if k not in team["form"]:
                    log.warning("[%s] form.%s em falta → usando %d", label, k, v)
                    team["form"][k] = v

        # Médias de golos da temporada
        if "avg_scored" not in team:
            log.warning("[%s] 'avg_scored' em falta → usando %.2f", label, DEFAULT_AVG_SCORED)
            team["avg_scored"] = DEFAULT_AVG_SCORED
        if "avg_conceded" not in team:
            log.warning("[%s] 'avg_conceded' em falta → usando %.2f", label, DEFAULT_AVG_CONCEDED)
            team["avg_conceded"] = DEFAULT_AVG_CONCEDED

        # Taxa de vitória em casa/fora
        if wr_key not in team:
            log.warning("[%s] '%s' em falta → usando %.2f", label, wr_key, default_wr)
            team[wr_key] = default_wr

        # Lesões/suspensões (número de ausências importantes)
        team.setdefault("injuries", 0)

    # Validação de ranges
    ha = d["home_advantage"]
    if not (0.9 <= ha <= 1.5):
        log.warning("home_advantage=%.2f fora do range esperado [0.9, 1.5].", ha)

    return d


# ─────────────────────────────────────────────────────────────────────────────
# MOTOR POISSON — Funções auxiliares
# ─────────────────────────────────────────────────────────────────────────────

def poisson_pmf(lam: float, k: int) -> float:
    """
    Calcula P(X = k) para X ~ Poisson(lam).

    Usa logaritmos para estabilidade numérica:
      log P(X=k) = k*log(lam) - lam - log(k!)
    """
    if lam <= 0.0:
        return 1.0 if k == 0 else 0.0
    log_factorial_k = sum(math.log(i) for i in range(1, k + 1)) if k > 0 else 0.0
    log_p = k * math.log(lam) - lam - log_factorial_k
    return math.exp(log_p)


def poisson_goal_probs(lam: float, max_k: int = MAX_K) -> list:
    """
    Retorna lista de probabilidades P(golos = k) para k = 0, 1, ..., max_k.

    O último elemento (k = max_k) acumula toda a massa restante:
      P(X >= max_k) = 1 - sum(P(X=0)..P(X=max_k-1))
    Isto garante que a distribuição soma exatamente 1.0.
    """
    probs = [poisson_pmf(lam, k) for k in range(max_k)]
    # Último bucket: probabilidade residual (golos >= max_k)
    residual = max(0.0, 1.0 - sum(probs))
    probs.append(residual)
    return probs  # comprimento = max_k + 1


def estimate_lambdas(data: Dict[str, Any]) -> Tuple[float, float]:
    """
    Estima golos esperados (lambda) para cada equipa usando:

      lambda_home_raw = atk_home × (def_away / league_avg) × home_advantage
      lambda_away_raw = atk_away × (def_home / league_avg)

    Fórmula dixon-coles simplificada:
      - atk_home  = média de golos marcados em casa (temporada)
      - def_away  = média de golos sofridos da equipa de fora
      - Dividir pela league_avg normaliza os ratings de ataque/defesa

    Shrinkage (evita extremos com poucos jogos):
      lambda = (1 - w) × raw + w × league_avg
      onde w = SHRINKAGE_WEIGHT (0.20)
    """
    home = data["home"]
    away = data["away"]
    ha   = data["home_advantage"]
    w    = SHRINKAGE_WEIGHT
    lg   = LEAGUE_AVG_GOALS

    atk_home  = home["avg_scored"]
    def_home  = home["avg_conceded"]
    atk_away  = away["avg_scored"]
    def_away  = away["avg_conceded"]

    # Lambdas brutos (sem shrinkage)
    raw_home = atk_home * (def_away / lg) * ha
    raw_away = atk_away * (def_home / lg)

    # Após shrinkage: puxa valores em direção à média da liga
    lam_home = (1.0 - w) * raw_home + w * lg
    lam_away = (1.0 - w) * raw_away + w * lg

    # Garantir valores positivos mínimos
    lam_home = max(0.1, lam_home)
    lam_away = max(0.1, lam_away)

    log.debug(
        "lambda: home=%.4f (raw=%.4f)  away=%.4f (raw=%.4f)  ha=%.2f",
        lam_home, raw_home, lam_away, raw_away, ha
    )
    return lam_home, lam_away


# ─────────────────────────────────────────────────────────────────────────────
# MOTOR POISSON — Principal
# ─────────────────────────────────────────────────────────────────────────────

def run_poisson(
    data: Dict[str, Any],
    include_optional: bool = True
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Motor Poisson — pipeline completo:

    1. Estimar lambda_home e lambda_away (golos esperados)
    2. Construir distribuições P(home_goals = i) e P(away_goals = j)
    3. Assumindo independência entre golos de cada equipa, calcular a
       matriz de resultados: P(i, j) = P(home=i) × P(away=j)
    4. Acumular:
         home_win = Σ P(i,j) para i > j
         draw     = Σ P(i,j) para i = j
         away_win = Σ P(i,j) para i < j
    5. Normalizar 1X2 para soma exata = 1.0
    6. (Opcional) calcular Over 2.5 e BTTS a partir da mesma matriz

    Retorna (probs_dict, debug_dict).
    """
    lam_home, lam_away = estimate_lambdas(data)

    p_home_goals = poisson_goal_probs(lam_home)
    p_away_goals = poisson_goal_probs(lam_away)
    n = len(p_home_goals)

    home_win = draw = away_win = 0.0
    over_2_5 = btts_yes = 0.0

    # Iterar sobre todas as combinações (i, j) de golos
    for i in range(n):
        for j in range(n):
            p_ij = p_home_goals[i] * p_away_goals[j]

            # 1X2
            if i > j:
                home_win += p_ij
            elif i == j:
                draw += p_ij
            else:
                away_win += p_ij

            if include_optional:
                # Over 2.5: total de golos >= 3
                if i + j >= 3:
                    over_2_5 += p_ij
                # BTTS: ambas as equipas marcam >= 1 golo
                if i >= 1 and j >= 1:
                    btts_yes += p_ij

    # Normalizar 1X2 (soma bruta pode ser ligeiramente < 1.0 devido a
    # truncamento do MAX_K — normalização fecha para 100%)
    total_1x2 = home_win + draw + away_win
    if total_1x2 > 0.0:
        home_win /= total_1x2
        draw     /= total_1x2
        away_win /= total_1x2

    log.debug(
        "1X2 (norm): home=%.4f draw=%.4f away=%.4f (soma_bruta=%.6f)",
        home_win, draw, away_win, total_1x2
    )

    result: Dict[str, float] = {
        "home_win": home_win,
        "draw":     draw,
        "away_win": away_win,
    }

    if include_optional:
        under_2_5 = max(0.0, 1.0 - over_2_5)
        btts_no   = max(0.0, 1.0 - btts_yes)
        result.update({
            "over_2_5":   over_2_5,
            "under_2_5":  under_2_5,
            "btts_yes":   btts_yes,
            "btts_no":    btts_no,
        })

    debug: Dict[str, Any] = {
        "lambda_home":          round(lam_home, 4),
        "lambda_away":          round(lam_away, 4),
        "home_advantage":       data["home_advantage"],
        "shrinkage_weight":     SHRINKAGE_WEIGHT,
        "league_avg_goals":     LEAGUE_AVG_GOALS,
        "sum_raw_1x2":          round(total_1x2, 6),
    }

    return result, debug


# ─────────────────────────────────────────────────────────────────────────────
# MOTOR HEURÍSTICO — Funções auxiliares
# ─────────────────────────────────────────────────────────────────────────────

def _form_score(form: Dict[str, int]) -> float:
    """
    Score de forma com base nos últimos 5 jogos (0..1).

    Componente 1 (70%): pontos obtidos / pontos máximos
      max = 5 vitórias × 3 pts = 15 pts
    Componente 2 (30%): diferença de golos normalizada para [0,1]
      gd in [-10, +10] → normalizado: (gd + 10) / 20
    """
    pts = form["wins"] * 3 + form["draws"] * 1
    pts_score = pts / 15.0
    pts_score = max(0.0, min(1.0, pts_score))

    gd = form.get("gf", 0) - form.get("ga", 0)
    gd_score = max(0.0, min(1.0, (gd + 10.0) / 20.0))

    return 0.70 * pts_score + 0.30 * gd_score


def _goals_score(avg_scored: float, avg_conceded: float) -> float:
    """
    Score de qualidade ofensiva/defensiva (0..1).

    rácio = golos_marcados / (golos_marcados + golos_sofridos)
    0.5 = equilíbrio perfeito, >0.5 = mais ofensivo que defensivo
    """
    total = avg_scored + avg_conceded
    if total <= 0.0:
        return 0.5
    return avg_scored / total


def _home_away_score(team: Dict[str, Any], side: str) -> float:
    """
    Score baseado na taxa de vitória em casa (para home) ou fora (para away).
    Retorna valor entre 0..1.
    """
    key = "home_win_rate" if side == "home" else "away_win_rate"
    return float(team.get(key, 0.45))


def _h2h_score(h2h: Optional[Dict[str, int]], side: str) -> float:
    """
    Score de head-to-head (0..1).
    Se não houver dados H2H, retorna 0.5 (neutro).

    Calcula fracção de vitórias do lado pedido sobre total de jogos.
    """
    if not h2h:
        return 0.5
    home_wins = h2h.get("home_wins", 0)
    away_wins = h2h.get("away_wins", 0)
    draws     = h2h.get("draws", 0)
    total = home_wins + away_wins + draws
    if total == 0:
        return 0.5
    return (home_wins / total) if side == "home" else (away_wins / total)


def _injury_penalty(injuries: int) -> float:
    """
    Penalidade por ausências importantes (0..0.20).
    Cada ausência retira 0.05 do score composto, com teto em 0.20.
    """
    return min(0.20, injuries * 0.05)


def _compute_team_score(
    team: Dict[str, Any],
    side: str,
    h2h: Optional[Dict[str, int]]
) -> float:
    """
    Calcula o score composto para uma equipa (aproximadamente 0..1).

    score = w_form × form_score
          + w_goals × goals_score
          + w_ha × home_away_score
          + w_h2h × h2h_score
          - injury_penalty

    Os pesos estão definidos em HEURISTIC_WEIGHTS.
    """
    w = HEURISTIC_WEIGHTS

    f  = _form_score(team["form"])
    g  = _goals_score(team["avg_scored"], team["avg_conceded"])
    ha = _home_away_score(team, side)
    h  = _h2h_score(h2h, side)

    score = (
        w["form"]      * f  +
        w["goals"]     * g  +
        w["home_away"] * ha +
        w["h2h"]       * h
    )

    penalty = _injury_penalty(team.get("injuries", 0))
    score = max(0.0, score - penalty)

    log.debug(
        "[%s] score=%.4f  form=%.4f goals=%.4f ha=%.4f h2h=%.4f penalty=%.4f",
        side, score, f, g, ha, h, penalty
    )
    return score


def _softmax3(a: float, b: float, c: float, temperature: float = 0.8) -> Tuple[float, float, float]:
    """
    Softmax sobre 3 valores com temperatura T.

    Temperatura < 1 → amplifica diferenças (distribuição mais "peaky")
    Temperatura > 1 → suaviza diferenças (distribuição mais uniforme)
    T = 0.8 dá amplificação moderada dos scores diferenciais.

    P_i = exp(v_i / T) / Σ exp(v_j / T)
    """
    exps = [math.exp(v / temperature) for v in (a, b, c)]
    total = sum(exps)
    return (exps[0] / total, exps[1] / total, exps[2] / total)


# ─────────────────────────────────────────────────────────────────────────────
# MOTOR HEURÍSTICO — Principal
# ─────────────────────────────────────────────────────────────────────────────

def run_heuristic(
    data: Dict[str, Any],
    include_optional: bool = True
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Motor heurístico — pipeline:

    1. Calcular score composto para home e away
    2. Ajustar score_home com multiplicador de vantagem casa
    3. Calcular score de empate como inverso da diferença
       (equipas muito equilibradas → maior probabilidade de empate)
    4. Softmax(score_home, score_draw, score_away) → distribuição inicial
    5. Aplicar piso de empate mínimo (MIN_DRAW_PCT)
    6. Normalizar para soma = 1.0
    7. (Opcional) estimar Over/Under e BTTS via Poisson simples

    Retorna (probs_dict, debug_dict).
    """
    home = data["home"]
    away = data["away"]
    h2h  = data.get("h2h")
    ha   = data["home_advantage"]

    score_home_raw = _compute_team_score(home, "home", h2h)
    score_away_raw = _compute_team_score(away, "away", h2h)

    # Aplicar vantagem em casa como multiplicador no score
    score_home_adj = score_home_raw * ha
    score_away_adj = score_away_raw

    log.debug("scores adj: home=%.4f away=%.4f", score_home_adj, score_away_adj)

    # Score de empate: maior quando equipas estão equilibradas
    # diff → 0 quando equipas iguais → draw_raw próximo de 0.6
    # diff → 1 quando muito diferentes → draw_raw pode ser negativo (→ 0)
    diff = abs(score_home_adj - score_away_adj)
    draw_raw = max(0.0, 0.60 - diff)

    # Softmax(home, draw, away) com temperatura 0.8
    p_home, p_draw, p_away = _softmax3(score_home_adj, draw_raw, score_away_adj)

    log.debug("pré-floor: home=%.4f draw=%.4f away=%.4f", p_home, p_draw, p_away)

    # Piso mínimo de empate: evita distribuições sem empate
    min_draw = MIN_DRAW_PCT / 100.0
    if p_draw < min_draw:
        deficit = min_draw - p_draw
        p_draw = min_draw
        # Redistribuir o deficit igualmente entre home e away
        p_home = max(0.0, p_home - deficit / 2.0)
        p_away = max(0.0, p_away - deficit / 2.0)

    # Normalização final (garante soma exata = 1.0)
    total = p_home + p_draw + p_away
    p_home /= total
    p_draw /= total
    p_away /= total

    log.debug("final: home=%.4f draw=%.4f away=%.4f", p_home, p_draw, p_away)

    result: Dict[str, float] = {
        "home_win": p_home,
        "draw":     p_draw,
        "away_win": p_away,
    }

    if include_optional:
        # Estimar Over/Under e BTTS usando Poisson com médias simples
        # (não usa toda a lógica de shrinkage do motor Poisson completo,
        #  mas dá uma estimativa razoável para o motor heurístico)
        lam_h = max(0.1, home["avg_scored"] * ha)
        lam_a = max(0.1, away["avg_scored"])
        ph = poisson_goal_probs(lam_h)
        pa = poisson_goal_probs(lam_a)
        n  = len(ph)

        over_2_5 = btts_yes = 0.0
        for i in range(n):
            for j in range(n):
                p_ij = ph[i] * pa[j]
                if i + j >= 3:
                    over_2_5 += p_ij
                if i >= 1 and j >= 1:
                    btts_yes += p_ij

        result.update({
            "over_2_5":  over_2_5,
            "under_2_5": max(0.0, 1.0 - over_2_5),
            "btts_yes":  btts_yes,
            "btts_no":   max(0.0, 1.0 - btts_yes),
        })

    debug: Dict[str, Any] = {
        "score_home_raw":    round(score_home_raw, 4),
        "score_away_raw":    round(score_away_raw, 4),
        "score_home_adj":    round(score_home_adj, 4),
        "score_away_adj":    round(score_away_adj, 4),
        "draw_raw":          round(draw_raw, 4),
        "home_advantage":    ha,
        "draw_floor_pct":    MIN_DRAW_PCT,
        "softmax_temp":      0.8,
    }

    return result, debug


# ─────────────────────────────────────────────────────────────────────────────
# FORMATAÇÃO DO OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def format_output(
    data: Dict[str, Any],
    probs: Dict[str, float],
    model_name: str,
    debug_info: Dict[str, Any],
    show_debug: bool
) -> Dict[str, Any]:
    """
    Constrói o dicionário JSON de saída.
    Converte probabilidades em percentagens arredondadas a 1 decimal.

    Verifica que 1X2 soma ~100% (tolerância ±0.5% por arredondamento).
    """
    def pct(v: float) -> float:
        return round(v * 100.0, 1)

    match_info: Dict[str, Any] = {
        "home": data["home"]["name"],
        "away": data["away"]["name"],
    }
    if data.get("date"):
        match_info["date"] = data["date"]

    probabilities: Dict[str, float] = {
        "home_win_pct": pct(probs["home_win"]),
        "draw_pct":     pct(probs["draw"]),
        "away_win_pct": pct(probs["away_win"]),
    }

    # Verificação interna: 1X2 deve somar 100%
    total_1x2 = (
        probabilities["home_win_pct"] +
        probabilities["draw_pct"]     +
        probabilities["away_win_pct"]
    )
    if abs(total_1x2 - 100.0) > 0.5:
        log.warning(
            "ATENÇÃO: 1X2 soma %.1f%% (esperado 100%%, diferença=%.2f%%)",
            total_1x2, abs(total_1x2 - 100.0)
        )

    # Mercados opcionais
    if "over_2_5" in probs:
        probabilities["over_2_5_pct"]  = pct(probs["over_2_5"])
        probabilities["under_2_5_pct"] = pct(probs["under_2_5"])
        probabilities["btts_yes_pct"]  = pct(probs["btts_yes"])
        probabilities["btts_no_pct"]   = pct(probs["btts_no"])

    output: Dict[str, Any] = {
        "match":       match_info,
        "model_used":  model_name,
        "probabilities": probabilities,
    }

    if show_debug:
        output["debug"] = debug_info

    return output


# ─────────────────────────────────────────────────────────────────────────────
# MOTOR NBA — Distribuição Normal
# ─────────────────────────────────────────────────────────────────────────────
#
# Basquetebol usa distribuição NORMAL (não Poisson) porque:
#   • Os pontos acumulam-se em ~100 posses por jogo → TCL aplica
#   • Diferenças de pontos aproximam-se a uma distribuição normal
#   • Scores típicos: 100–130 pts/equipa → variância estável
#
# Modelo:
#   1. Estimar pts_home e pts_away (pontos esperados)
#      pts_home = atk_home × (def_away / liga_avg) + home_adv
#      pts_away = atk_away × (def_home / liga_avg)
#   2. P(home win) = Φ(diff / σ_diff)  via função de erro (erf)
#   3. Over/Under para várias linhas  = Φ((total - linha) / σ_total)
# ─────────────────────────────────────────────────────────────────────────────

NBA_LEAGUE_AVG_PTS: float = 115.0   # média de pontos por equipa por jogo
NBA_HOME_ADV_PTS:   float = 3.0     # vantagem de casa em pontos (~3 pts NBA)
NBA_SCORE_STD:      float = 13.0    # desvio padrão de pontos de uma equipa
NBA_DIFF_STD:       float = 14.0    # desvio padrão da diferença (usado para win%)
NBA_SHRINKAGE:      float = 0.15    # 15% regressão à média da liga
NBA_DEFAULT_AVG:    float = 115.0   # default se faltar avg_scored/conceded
NBA_OU_LINES: list  = [210.5, 215.5, 220.5, 225.5, 230.5, 235.5]

# Defaults de forma NBA
NBA_DEFAULT_FORM: Dict[str, int] = {
    "wins": 2, "draws": 0, "losses": 3, "gf": 575, "ga": 580
}


def _norm_cdf(x: float) -> float:
    """
    CDF da distribuição normal padrão usando a função de erro (erf).
    Φ(x) = 0.5 × (1 + erf(x / √2))
    Usa apenas math da stdlib — sem numpy/scipy.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def apply_defaults_nba(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preenche campos em falta com defaults NBA.
    Emite log.warning para cada campo omitido.
    """
    d = dict(raw)
    d.setdefault("date", None)
    d.setdefault("h2h", None)
    d.setdefault("home_advantage_pts", NBA_HOME_ADV_PTS)

    for side, label, wr_key, default_wr in [
        ("home", "casa", "home_win_rate", 0.60),
        ("away", "fora", "away_win_rate", 0.40),
    ]:
        if side not in d or not isinstance(d.get(side), dict):
            log.warning("[NBA][%s] não definida → defaults.", label)
            d[side] = {}
        team = d[side]
        team.setdefault("name", f"Equipa {label.capitalize()}")

        if "form" not in team or not isinstance(team.get("form"), dict):
            log.warning("[NBA][%s] form em falta → defaults.", label)
            team["form"] = dict(NBA_DEFAULT_FORM)
        else:
            for k, v in NBA_DEFAULT_FORM.items():
                if k not in team["form"]:
                    team["form"][k] = v

        if "avg_scored" not in team:
            log.warning("[NBA][%s] avg_scored em falta → %.1f", label, NBA_DEFAULT_AVG)
            team["avg_scored"] = NBA_DEFAULT_AVG
        if "avg_conceded" not in team:
            log.warning("[NBA][%s] avg_conceded em falta → %.1f", label, NBA_DEFAULT_AVG)
            team["avg_conceded"] = NBA_DEFAULT_AVG
        if wr_key not in team:
            team[wr_key] = default_wr
        team.setdefault("injuries", 0)

    return d


def run_nba(data: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Motor NBA — pipeline:

    1. Estimar pontos esperados por equipa
         pts_home_raw = atk_home × (def_away / liga_avg) + home_adv
         pts_away_raw = atk_away × (def_home / liga_avg)
    2. Shrinkage 15% → suaviza quando poucos jogos
    3. Win probability via distribuição normal:
         diff = pts_home - pts_away
         P(home win) = Φ(diff / σ_diff)
         P(away win) = 1 - P(home win)
         (NBA não tem empate — prolongamento decide)
    4. Over/Under para as linhas standard NBA:
         σ_total = √(σ_home² + σ_away²) ≈ √2 × NBA_SCORE_STD
         P(over L) = 1 - Φ((L - total_expected) / σ_total)
    5. Ajuste de lesões: cada lesão subtrai 1 pto esperado (máx 4)

    Retorna (probs_dict, debug_dict).
    """
    home  = data["home"]
    away  = data["away"]
    ha    = data.get("home_advantage_pts", NBA_HOME_ADV_PTS)
    w     = NBA_SHRINKAGE
    lg    = NBA_LEAGUE_AVG_PTS

    atk_h = home["avg_scored"]
    def_h = home["avg_conceded"]
    atk_a = away["avg_scored"]
    def_a = away["avg_conceded"]

    # Pontos esperados brutos
    raw_h = atk_h * (def_a / lg) + ha
    raw_a = atk_a * (def_h / lg)

    # Shrinkage
    pts_home = (1.0 - w) * raw_h + w * lg
    pts_away = (1.0 - w) * raw_a + w * lg

    # Ajuste de lesões (cada ausência chave → -1 pt, teto 4)
    pts_home -= min(4.0, home.get("injuries", 0) * 1.0)
    pts_away -= min(4.0, away.get("injuries", 0) * 1.0)
    pts_home = max(80.0, pts_home)
    pts_away = max(80.0, pts_away)

    # Win probability (normal)
    diff        = pts_home - pts_away
    p_home_win  = _norm_cdf(diff / NBA_DIFF_STD)
    p_away_win  = 1.0 - p_home_win

    # Over/Under para linhas standard
    total_exp  = pts_home + pts_away
    sigma_tot  = math.sqrt(2.0) * NBA_SCORE_STD   # ≈ 18.4 pts
    ou_probs: Dict[str, float] = {}
    for line in NBA_OU_LINES:
        p_over  = 1.0 - _norm_cdf((line - total_exp) / sigma_tot)
        p_under = 1.0 - p_over
        key = str(line).replace(".", "_")
        ou_probs[f"over_{key}"]  = max(0.001, min(0.999, p_over))
        ou_probs[f"under_{key}"] = max(0.001, min(0.999, p_under))

    result: Dict[str, float] = {
        "home_win": p_home_win,
        "away_win": p_away_win,
        "draw":     0.0,           # NBA não tem empate
        "pts_expected_home": pts_home,
        "pts_expected_away": pts_away,
        "total_expected":    total_exp,
        **ou_probs,
    }

    debug: Dict[str, Any] = {
        "pts_expected_home": round(pts_home, 2),
        "pts_expected_away": round(pts_away, 2),
        "diff":              round(diff, 2),
        "total_expected":    round(total_exp, 2),
        "sigma_diff":        NBA_DIFF_STD,
        "sigma_total":       round(sigma_tot, 2),
        "home_advantage_pts": ha,
        "shrinkage":         NBA_SHRINKAGE,
    }

    log.debug(
        "[NBA] pts home=%.1f away=%.1f  diff=%.1f  P(home)=%.3f",
        pts_home, pts_away, diff, p_home_win
    )

    return result, debug


def format_output_nba(
    data: Dict[str, Any],
    probs: Dict[str, float],
    debug_info: Dict[str, Any],
    show_debug: bool
) -> Dict[str, Any]:
    """
    Formata saída JSON para NBA.
    Inclui: moneyline (home/away), pontos esperados, Over/Under por linha.
    """
    def pct(v: float) -> float:
        return round(v * 100.0, 1)

    match_info: Dict[str, Any] = {
        "home": data["home"]["name"],
        "away": data["away"]["name"],
    }
    if data.get("date"):
        match_info["date"] = data["date"]

    probabilities: Dict[str, Any] = {
        "home_win_pct":       pct(probs["home_win"]),
        "away_win_pct":       pct(probs["away_win"]),
        "pts_expected_home":  round(probs["pts_expected_home"], 1),
        "pts_expected_away":  round(probs["pts_expected_away"], 1),
        "total_expected":     round(probs["total_expected"], 1),
        "over_under_lines":   {},
    }

    for line in NBA_OU_LINES:
        key = str(line).replace(".", "_")
        probabilities["over_under_lines"][str(line)] = {
            "over_pct":  pct(probs.get(f"over_{key}",  0.5)),
            "under_pct": pct(probs.get(f"under_{key}", 0.5)),
        }

    output: Dict[str, Any] = {
        "match":         match_info,
        "sport":         "nba",
        "model_used":    "nba_normal",
        "probabilities": probabilities,
    }
    if show_debug:
        output["debug"] = debug_info

    return output


# Demo data NBA
NBA_DEMO_DATA: Dict[str, Any] = {
    "home": {
        "name": "Boston Celtics",
        "form": {"wins": 4, "draws": 0, "losses": 1, "gf": 608, "ga": 565},
        "avg_scored":    120.3,
        "avg_conceded":  110.2,
        "home_win_rate": 0.72,
        "injuries":      0,
    },
    "away": {
        "name": "Golden State Warriors",
        "form": {"wins": 2, "draws": 0, "losses": 3, "gf": 565, "ga": 590},
        "avg_scored":    113.8,
        "avg_conceded":  116.5,
        "away_win_rate": 0.42,
        "injuries":      1,
    },
    "date":                "2026-03-10",
    "home_advantage_pts":  3.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# DADOS DE DEMONSTRAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

DEMO_DATA: Dict[str, Any] = {
    "home": {
        "name": "Sporting CP",
        "form": {
            "wins": 4, "draws": 0, "losses": 1,
            "gf": 11, "ga": 4
        },
        "avg_scored":    2.10,
        "avg_conceded":  0.90,
        "home_win_rate": 0.72,
        "injuries":      1
    },
    "away": {
        "name": "FC Porto",
        "form": {
            "wins": 2, "draws": 2, "losses": 1,
            "gf": 7, "ga": 6
        },
        "avg_scored":    1.60,
        "avg_conceded":  1.20,
        "away_win_rate": 0.45,
        "injuries":      2
    },
    "h2h": {
        "home_wins": 3,
        "away_wins": 2,
        "draws":     1
    },
    "date":           "2026-03-10",
    "home_advantage": 1.10
}


# ─────────────────────────────────────────────────────────────────────────────
# TESTES DE SANIDADE
# ─────────────────────────────────────────────────────────────────────────────

def run_tests() -> None:
    """
    Testes automáticos de sanidade:

    1. 1X2 soma 100% para ambos os motores (dados completos)
    2. Nenhuma probabilidade é negativa
    3. Motor funciona com dados mínimos (apenas nomes)
    4. Poisson PMF soma ~1.0
    5. Motor heurístico respeita o piso de empate
    """
    setup_logging(debug=False)
    print("=" * 50)
    print("A correr testes de sanidade…")
    print("=" * 50)
    errors: list = []

    # --- Teste 1 & 2: dados completos, ambos os motores ---
    for model_name, run_fn in [("poisson", run_poisson), ("heuristic", run_heuristic)]:
        d = apply_defaults(dict(DEMO_DATA))
        probs, dbg = run_fn(d, include_optional=True)

        total = probs["home_win"] + probs["draw"] + probs["away_win"]
        ok_sum = abs(total - 1.0) < 1e-9

        for k, v in probs.items():
            if v < -1e-9:
                errors.append(f"[{model_name}] probabilidade negativa: {k}={v:.6f}")

        status = "OK" if ok_sum else f"FALHA (soma={total:.9f})"
        print(
            f"  [{model_name}] home={probs['home_win']:.3f}  "
            f"draw={probs['draw']:.3f}  away={probs['away_win']:.3f}  "
            f"soma={total:.9f}  → {status}"
        )
        if not ok_sum:
            errors.append(f"[{model_name}] 1X2 não soma 1.0 (soma={total:.9f})")

    # --- Teste 3: dados mínimos ---
    for model_name, run_fn in [("poisson", run_poisson), ("heuristic", run_heuristic)]:
        minimal = {"home": {"name": "A"}, "away": {"name": "B"}}
        dm = apply_defaults(minimal)
        probs2, _ = run_fn(dm, include_optional=True)
        total2 = probs2["home_win"] + probs2["draw"] + probs2["away_win"]
        ok2 = abs(total2 - 1.0) < 1e-9
        status2 = "OK" if ok2 else f"FALHA (soma={total2:.9f})"
        print(f"  [{model_name}][minimal] soma={total2:.9f}  → {status2}")
        if not ok2:
            errors.append(f"[{model_name}][minimal] soma != 1.0")

    # --- Teste 4: Poisson PMF soma ~1.0 ---
    for lam in [0.5, 1.35, 2.5, 4.0]:
        probs_g = poisson_goal_probs(lam)
        total_g = sum(probs_g)
        ok_g = abs(total_g - 1.0) < 1e-9
        status_g = "OK" if ok_g else f"FALHA ({total_g:.9f})"
        print(f"  [pmf] lam={lam}  soma={total_g:.9f}  → {status_g}")
        if not ok_g:
            errors.append(f"[pmf] lam={lam} soma != 1.0")

    # --- Teste 5: piso de empate no heurístico ---
    d = apply_defaults(dict(DEMO_DATA))
    # Forçar uma equipa muito superior para testar o piso
    d["home"]["avg_scored"]   = 5.0
    d["home"]["avg_conceded"] = 0.1
    d["home"]["form"]         = {"wins": 5, "draws": 0, "losses": 0, "gf": 20, "ga": 0}
    d["home"]["home_win_rate"] = 0.99
    probs_floor, _ = run_heuristic(d, include_optional=False)
    draw_pct = probs_floor["draw"] * 100.0
    ok_floor = draw_pct >= (MIN_DRAW_PCT - 0.01)
    print(
        f"  [heuristic][floor] draw={draw_pct:.1f}%  "
        f"(mínimo={MIN_DRAW_PCT}%)  → {'OK' if ok_floor else 'FALHA'}"
    )
    if not ok_floor:
        errors.append(f"[heuristic] draw {draw_pct:.1f}% < floor {MIN_DRAW_PCT}%")

    print("=" * 50)
    if errors:
        print(f"FALHOU {len(errors)} teste(s):")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print(f"Todos os testes passaram  ✓  ({5} grupos de testes)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="predictor.py",
        description=(
            "Preditor de probabilidades de futebol.\n"
            "Gera percentagens 1X2, Over/Under 2.5 e BTTS.\n"
            "Sem conselhos de aposta — apenas probabilidades."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Exemplos:\n"
            "  python predictor.py --model poisson   --input game.json\n"
            "  python predictor.py --model heuristic --demo --debug\n"
            "  python predictor.py --test\n"
            "  cat game.json | python predictor.py --model poisson --input -"
        ),
    )

    parser.add_argument(
        "--model",
        choices=["poisson", "heuristic"],
        default="poisson",
        help=(
            "Motor de previsão (default: poisson):\n"
            "  poisson   — distribuição de Poisson via golos esperados\n"
            "  heuristic — pesos heurísticos com softmax"
        ),
    )
    parser.add_argument(
        "--input",
        metavar="FILE",
        help="Ficheiro JSON com dados do jogo ('-' para ler de stdin).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Usa dados fictícios de demonstração (Sporting CP vs FC Porto).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Inclui campo 'debug' com valores intermédios no output JSON.",
    )
    parser.add_argument(
        "--no-optional",
        action="store_true",
        dest="no_optional",
        help="Omite mercados Over/Under 2.5 e BTTS do output.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Corre os testes de sanidade internos.",
    )

    args = parser.parse_args()
    setup_logging(debug=args.debug)

    # Modo de testes
    if args.test:
        run_tests()
        return

    # Carregar dados
    if args.demo:
        raw = dict(DEMO_DATA)

    elif args.input:
        try:
            if args.input == "-":
                raw = json.load(sys.stdin)
            else:
                with open(args.input, "r", encoding="utf-8") as f:
                    raw = json.load(f)
        except FileNotFoundError:
            print(
                f"Erro: ficheiro '{args.input}' não encontrado.",
                file=sys.stderr
            )
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Erro: JSON inválido — {e}", file=sys.stderr)
            sys.exit(1)

    else:
        parser.print_help()
        print(
            "\nErro: especifica --input FILE (ou '-' para stdin) ou usa --demo.",
            file=sys.stderr
        )
        sys.exit(1)

    # Aplicar defaults
    data = apply_defaults(raw)

    # Correr o motor escolhido
    include_optional = not args.no_optional
    if args.model == "poisson":
        probs, debug_info = run_poisson(data, include_optional=include_optional)
    else:
        probs, debug_info = run_heuristic(data, include_optional=include_optional)

    # Formatar e imprimir output JSON
    output = format_output(
        data, probs, args.model, debug_info, show_debug=args.debug
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
