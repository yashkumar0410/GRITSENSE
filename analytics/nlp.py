"""
NLP backends that turn structured match facts into live commentary.

    LocalLLMBackend - small instruction-tuned model via Hugging Face transformers (CPU OK)

Commentary is fully offline: the model is loaded from the local Hugging Face cache
and only downloaded if it has never been fetched before. `build_backend` returns
None when the model can't be loaded, and the commentary generator falls back to its
templates. A backend that fails at runtime disables itself so rendering never stalls.
"""

from __future__ import annotations

import json

SYSTEM_PROMPT = (
    "You are a live TV volleyball commentator with a tactical analyst's eye. "
    "From the structured match state you are given, write exactly ONE sentence of live "
    "commentary, at most 32 words. Say what the attacking team should play and where, "
    "using the player number, the zone and one supporting number (point chance, block chance "
    "or the gap in metres). Sound energetic and natural, vary your phrasing, do not "
    "repeat the recent lines, and output only the sentence - no quotes, no preamble."
)

DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def build_prompt(kind: str, facts: dict, recent: list[str]) -> str:
    event = {
        "possession": "The ball has just crossed the net; possession changed.",
        "move": "The tactical model has a new recommended move.",
        "insight": "No new event; give a tactical insight about the key matchup or team shape.",
    }[kind]
    return (
        f"Event: {event}\n"
        f"Match state (JSON):\n{json.dumps(facts, indent=1)}\n"
        f"Recent commentary (do not repeat):\n" + "\n".join(f"- {r}" for r in recent[-3:])
    )


_EVENT_TEXT = {
    "possession": "the ball just crossed the net",
    "move": "new recommended move",
    "insight": "tactical insight",
}


def facts_to_text(kind: str, facts: dict) -> str:
    """Plain-language fact sheet; easier than JSON for small local models."""
    matchup = facts.get("key_matchup")
    shape = facts.get("team_shape", {})
    lines = [
        f"Event: {_EVENT_TEXT[kind]}.",
        f"{facts['attacking_team']} attacking, {facts['defending_team']} defending.",
        f"Move: {facts['recommended_move'].lower()} by {facts['key_player']} ({facts['move_detail']}).",
        f"Target: Zone {facts['target_zone']}, {facts['target_description']}.",
        f"Point chance {facts['epv']:.0%}, block chance {facts['block_chance']}, "
        f"gap {facts['gap_to_nearest_defender_m']} m.",
    ]
    if matchup:
        lines.append(f"Key matchup: {matchup['attacker']} against defender {matchup['defender']}, "
                     f"who removes {matchup['share_of_threat_removed']} of the threat.")
    if shape:
        lines.append(f"{facts['defending_team']} can reach {shape.get('defending_coverage')} of their court in time.")
    return "\n".join(lines)


_EXAMPLE_FACTS = facts_to_text("move", {
    "attacking_team": "ITA", "defending_team": "POL", "recommended_move": "SPIKE · CROSS-COURT",
    "move_detail": "high hands, sharp angle", "key_player": "#7", "target_zone": 1,
    "target_description": "deep right corner", "epv": 0.66, "block_chance": "8%",
    "gap_to_nearest_defender_m": 3.1, "key_matchup": None, "team_shape": {},
})
_EXAMPLE_LINE = ("#7 has the angle for ITA - rip that cross-court spike into the deep right "
                 "corner, Zone 1, where POL have left 3.1 metres open!")


def is_grounded(text: str, facts: dict) -> bool:
    """Reject generations that drop the key facts (zone number and player)."""
    zone_ok = str(facts.get("target_zone", "")) in text
    player_ok = str(facts.get("key_player", "#")).lstrip("#") in text
    return zone_ok and player_ok


def _clean(text: str) -> str:
    text = " ".join(text.strip().split())
    return text.strip('"').strip("'").strip()


class LocalLLMBackend:
    name = "LOCAL LLM"

    def __init__(self, model: str = DEFAULT_LOCAL_MODEL):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import pipeline

        try:
            # Offline: resolve the cached copy without touching the network.
            path = snapshot_download(model, local_files_only=True)
            print(f"  [nlp] loading local model {model} from cache (offline)")
        except Exception:
            print(f"  [nlp] {model} not cached yet - downloading once…")
            path = snapshot_download(model)
        # GPU when there is one; "auto" dtype keeps the checkpoint's half precision.
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("text-generation", model=path, device=device, dtype="auto")

    def generate(self, kind: str, facts: dict, recent: list[str]) -> str | None:
        # Small models copy whatever text they are shown, so they get plain facts and
        # one example instead of JSON plus the recent lines.
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _EXAMPLE_FACTS},
            {"role": "assistant", "content": _EXAMPLE_LINE},
            {"role": "user", "content": facts_to_text(kind, facts)},
        ]
        out = self.pipe(messages, max_new_tokens=60, do_sample=True, temperature=0.7, top_p=0.9)
        text = out[0]["generated_text"][-1]["content"]
        # Keep only the first sentence; small models sometimes ramble.
        for stop in (". ", "! ", "? "):
            if stop in text:
                text = text.split(stop)[0] + stop.strip()
                break
        return _clean(text) or None


def build_backend(mode: str = "local", local_model: str = DEFAULT_LOCAL_MODEL):
    """Return the offline local backend, or None for template-only commentary."""
    if mode == "template":
        return None
    try:
        return LocalLLMBackend(local_model)
    except ImportError:
        print("  [nlp] local LLM unavailable - transformers not installed (pip install transformers)")
    except Exception as exc:
        print(f"  [nlp] local LLM unavailable - {type(exc).__name__}: {str(exc)[:120]}")
    return None
