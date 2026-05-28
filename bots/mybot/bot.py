"""
╔══════════════════════════════════════════════════════════════╗
║         FULLHOUSE HACKATHON — BOT TEMPLATE v1.0             ║
║         No-Limit Texas Hold'em, 6-max                        ║
╚══════════════════════════════════════════════════════════════╝

RULES:
  - Implement the decide() function below. That's it.
  - You may import any stdlib module and any library in requirements.txt
  - You have 2 seconds to return an action or you auto-fold
  - If your function crashes, it auto-folds for that hand

NOT ALLOWED (will DQ your bot):
  - External API calls: no Claude/OpenAI/Anthropic/Google/any HTTP. Network is
    blocked at the container level; trying anyway is a DQ.
  - File writes during gameplay; data/ is read-only and only at import time.
  - subprocess / os.system / shell commands.
  - Threading or async tricks to dodge the 2s/action signal timer.
  - Reflection: __import__('socket'), getattr(__builtins__, 'open'),
    eval(), exec(), compile() — all flagged by the validator.
  - Collusion between bots you've registered with friends — bots must play
    independently; coordinated soft-play or chip-dumping = both DQ'd.
  - Reading other bots' code or hole cards (you can't anyway, but trying = DQ).

OPTIONAL DATA FILES (NEW):
  Submit a .zip archive containing:
    bot.py        (this file, required at root)
    data/         (optional directory with .npz, .pkl, .bin, etc.)

  At module-import time only, you can read from a sibling 'data/' directory:

      import os
      DATA_DIR = os.environ.get("BOT_DATA_DIR",
                                os.path.join(os.path.dirname(__file__), "data"))
      with open(os.path.join(DATA_DIR, "blueprint.npz"), "rb") as f:
          BLUEPRINT = ...load(f)

  Limits:
    - Total submission (bot.py + data/) <= 250 MB
    - data/ alone <= 200 MB
    - bot.py <= 5 MB
    - File access during decide() is blocked at the OS level

CARD FORMAT:
  Cards are strings like "As" (Ace of spades), "Td" (Ten of diamonds)
  Ranks: 2 3 4 5 6 7 8 9 T J Q K A
  Suits: s (spades) h (hearts) d (diamonds) c (clubs)

RETURN FORMAT:
  {"action": "fold"}
  {"action": "check"}          # only valid when amount_owed == 0
  {"action": "call"}
  {"action": "raise", "amount": 1200}   # amount = TOTAL bet, not raise-by
  {"action": "all_in"}

  Invalid actions default to fold. Raises below min_raise_to are snapped up.
"""

import eval7
import random

BOT_NAME   = "XYZ-V2"
BOT_AVATAR = "robot_2"

# Card lookup built at import time — avoids constructing eval7.Card per simulation.
_CARD = {r + s: eval7.Card(r + s) for r in "23456789TJQKA" for s in "shdc"}
_DECK = list(_CARD.keys())   # 52 card strings in fixed order

_RANK_VAL = {r: i for i, r in enumerate("23456789TJQKA")}


# ── Preflop hand scoring (for range-aware MC) ─────────────────────────────────

def _hand_score(c1: str, c2: str) -> float:
    """Fast preflop hand strength heuristic. Higher = stronger."""
    r1, r2 = _RANK_VAL[c1[0]], _RANK_VAL[c2[0]]
    if r1 < r2:
        r1, r2 = r2, r1
    suited = c1[1] == c2[1]
    if r1 == r2:
        return 8.0 + r1              # pairs: 8–20
    score = r1 + r2 * 0.5
    gap   = r1 - r2
    if suited: score += 2.0
    if gap <= 1: score += 1.5
    elif gap <= 2: score += 0.7
    return score

# Precompute sorted combo scores so _score_threshold() is O(1).
_ALL_SCORES = sorted(
    _hand_score(_DECK[i], _DECK[j])
    for i in range(52) for j in range(i + 1, 52)
)  # 1326 entries

def _score_threshold(top_fraction: float) -> float:
    """Min _hand_score to be in the top `top_fraction` of starting hands."""
    idx = max(0, int((1.0 - top_fraction) * len(_ALL_SCORES)))
    return _ALL_SCORES[min(idx, len(_ALL_SCORES) - 1)]


# ── Equity estimation ─────────────────────────────────────────────────────────

def _estimate_equity(hole: list, board: list, n_opp: int, n_sims: int = 400,
                     opp_range: float = 1.0) -> float:
    """
    Monte Carlo equity vs n_opp opponents.
    opp_range: fraction of starting hands opponents are assumed to hold
               (1.0 = any two cards; 0.30 = top 30%).  Hands outside the
               range are skipped during sampling (burn-and-replace style).
    """
    if n_opp <= 0:
        return 1.0

    dead        = set(hole + board)
    live        = [c for c in _DECK if c not in dead]   # strings
    hole_cards  = [_CARD[c] for c in hole]
    board_cards = [_CARD[c] for c in board]
    to_come     = 5 - len(board_cards)

    if len(live) < n_opp * 2 + to_come:
        return 0.5

    min_score = _score_threshold(opp_range) if opp_range < 0.95 else -float("inf")

    wins = 0.0
    sims_done = 0
    for _ in range(n_sims):
        random.shuffle(live)
        idx       = 0
        opp_hands = []
        valid     = True

        for _ in range(n_opp):
            found = False
            while idx + 1 < len(live):
                if _hand_score(live[idx], live[idx + 1]) >= min_score:
                    found = True
                    break
                idx += 2          # burn this pair; not in range
            if not found:
                valid = False
                break
            opp_hands.append([_CARD[live[idx]], _CARD[live[idx + 1]]])
            idx += 2

        if not valid or idx + to_come > len(live):
            continue

        run_board = board_cards + [_CARD[c] for c in live[idx: idx + to_come]]
        my_score  = eval7.evaluate(hole_cards + run_board)
        best_opp  = max(eval7.evaluate(h + run_board) for h in opp_hands)
        if   my_score > best_opp: wins += 1.0
        elif my_score == best_opp: wins += 0.5
        sims_done += 1

    return wins / sims_done if sims_done > 0 else 0.5


# ── Position ──────────────────────────────────────────────────────────────────

def _position_score(gs: dict) -> float:
    """
    Returns position quality in [0.0, 1.0].
      1.0  = dealer / BTN  (acts last postflop — best)
      0.0  = SB            (acts first postflop — worst)

    Postflop position order (best → worst): BTN > CO > … > BB > SB.
    We infer the dealer seat from the small-blind entry in action_log.
    """
    n = len(gs["players"])
    if n <= 1:
        return 1.0

    sb_seat = None
    for e in gs["action_log"]:
        if e["action"] == "small_blind":
            sb_seat = e["seat"]
            break
    if sb_seat is None:
        return 0.5

    # Heads-up: dealer == SB.  Multi-way: dealer is one seat before SB.
    dealer = sb_seat if n == 2 else (sb_seat - 1) % n
    # dist: seats clockwise from dealer.  0 = BTN, 1 = SB, 2 = BB, …, n-1 = CO
    dist = (gs["seat_to_act"] - dealer) % n

    if dist == 0:
        return 1.0                      # BTN — best
    return (dist - 1) / (n - 1)        # SB=0.0  BB=1/(n-1)  …  CO=(n-2)/(n-1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _n_opp(gs: dict) -> int:
    my = gs["seat_to_act"]
    return sum(1 for p in gs["players"] if p["seat"] != my and not p["is_folded"])


def _infer_opp_range(gs: dict) -> float:
    """
    Estimate the fraction of starting hands opponents are likely holding.
    Uses observed aggression as a proxy for range tightness.
    Preflop raises narrow range significantly; postflop bets narrow it
    moderately.  Defaults to 0.60 (wide but not random) when uncontested.
    """
    street  = gs["street"]
    cur_bet = gs["current_bet"]
    owed    = gs["amount_owed"]
    pot     = gs["pot"]

    if street == "preflop":
        if cur_bet > 400:   return 0.12   # 4-bet range
        if cur_bet > 200:   return 0.18   # 3-bet range
        if cur_bet > 100:   return 0.28   # open-raise range
        return 0.60                        # limped / blind-vs-blind

    # Postflop: tighten based on bet size relative to pot
    if owed > 0 and pot > 0:
        if owed / pot >= 0.60:  return 0.28   # large bet
        return 0.40                            # smaller bet
    return 0.55                                # checked to us


# ── Main decision function ────────────────────────────────────────────────────

def decide(game_state: dict) -> dict:
    """
    Called once per action. Must return within 2 seconds.

    game_state keys:
      hand_id          str   — unique hand identifier
      street           str   — "preflop" | "flop" | "turn" | "river"
      seat_to_act      int   — your seat number (0-5)
      pot              int   — total chips in pot
      community_cards  list  — e.g. ["As", "Kd", "7h"] (empty preflop)
      current_bet      int   — highest bet on this street
      min_raise_to     int   — minimum legal raise total
      amount_owed      int   — chips you need to put in to call (0 = free check)
      can_check        bool  — True when amount_owed == 0
      your_cards       list  — your two hole cards, e.g. ["Ah", "Kh"]
      your_stack       int   — your remaining chips
      your_bet_this_street int — chips you've already put in this street
      players          list  — public info on all seats (see below)
      action_log       list  — all actions so far this hand

    players[i] keys (public info only, no hole cards):
      seat, bot_id, stack, is_active, is_folded, is_all_in, bet_this_street
    """

    hole      = game_state["your_cards"]
    board     = game_state["community_cards"]
    pot       = game_state["pot"]
    owed      = game_state["amount_owed"]
    stack     = game_state["your_stack"]
    min_r     = game_state["min_raise_to"]
    bts       = game_state["your_bet_this_street"]
    can_check = game_state["can_check"]
    cur_bet   = game_state["current_bet"]
    street    = game_state["street"]

    pos     = _position_score(game_state)         # 0.0 (SB) → 1.0 (BTN)
    n_opp   = _n_opp(game_state)
    opp_rng = _infer_opp_range(game_state)
    eq      = _estimate_equity(hole, board, n_opp, opp_range=opp_rng)

    # Relative equity: >1.0 means our hand is above the expected average for
    # a random hand facing n_opp opponents.
    exp_eq = 1.0 / (n_opp + 1) if n_opp > 0 else 1.0
    rel_eq = eq / exp_eq if exp_eq > 0 else 1.0

    # Minimum equity needed to break even on a call.
    pot_odds = owed / (pot + owed) if owed > 0 and (pot + owed) > 0 else 0.0

    def make_raise(frac: float) -> dict:
        """Raise to (current_bet + frac * pot), clamped to the legal range."""
        total = cur_bet + int(pot * frac)
        total = max(total, min_r)
        total = min(total, stack + bts)
        return {"action": "raise", "amount": total}

    # ── Thresholds (in relative-equity units) ────────────────────────────────
    # Late position (pos→1) lowers thresholds so the bot plays a wider range.
    #   raise_thr:  BTN=1.30  SB=1.80
    raise_thr = 1.8 - pos * 0.5

    # Absolute call threshold: must beat pot odds by a position-scaled margin,
    # and must clear a minimum relative-equity floor.
    call_margin = 0.06 - pos * 0.04            # BTN: 0.02   SB: 0.06
    call_thr    = max(pot_odds + call_margin,
                      (1.3 - pos * 0.3) * exp_eq)

    # ── Preflop ──────────────────────────────────────────────────────────────
    if street == "preflop":
        if rel_eq >= raise_thr:
            if cur_bet <= 100:          # open raise: 2.5–4 BB based on position
                total = int(100 * (2.5 + pos * 1.5))
            else:                       # 3-bet: ~3x the aggressor's bet
                total = cur_bet * 3
            total = max(total, min_r)
            total = min(total, stack + bts)
            return {"action": "raise", "amount": total}

        if can_check:
            return {"action": "check"}

        if eq >= call_thr:
            return {"action": "call"}

        return {"action": "fold"}

    # ── Postflop (flop / turn / river) ────────────────────────────────────────
    if rel_eq >= raise_thr:
        frac = min(0.75, 0.40 + (rel_eq - raise_thr) * 0.5)
        return make_raise(frac)

    if can_check:
        if pos >= 0.65:
            if rel_eq >= 1.15:
                # Thin value / semi-bluff probe
                return make_raise(0.35)
            # Pure bluff — scale frequency down with more opponents in the pot.
            # River: 30 %, flop: 22 %, turn: 15 % (heads-up); halved per extra opp.
            base = {"river": 0.30, "flop": 0.22, "turn": 0.15}.get(street, 0)
            bluff_prob = base * max(0.0, 1.0 - (n_opp - 1) * 0.45)
            if bluff_prob > 0 and random.random() < bluff_prob:
                bluff_size = {"river": 0.65, "flop": 0.35, "turn": 0.45}.get(street, 0.40)
                return make_raise(bluff_size)
        return {"action": "check"}

    if eq >= call_thr:
        return {"action": "call"}

    return {"action": "fold"}
