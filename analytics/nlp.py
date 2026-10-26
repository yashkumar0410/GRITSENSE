"""
NLP backends that turn structured match facts into live commentary.

    ClaudeBackend   - Claude via the Anthropic SDK (needs ANTHROPIC_API_KEY or `ant auth login`)
    LocalLLMBackend - small instruction-tuned model via Hugging Face transformers (CPU OK)

`build_backend("auto")` tries Claude, then the local model; it returns None when
neither is usable, and the commentary generator falls back to its templates.
A backend that fails at runtime disables itself so rendering never stalls.
"""

from __future__ import annotations

import json
import os

SYSTEM_PROMPT = (
    "You are a live TV volleyball commentator with a tactical analyst's eye. "
    "From the structured match state you are given, write exactly ONE sentence of live "
    "commentary, at most 32 words. Say what the attacking team should play and where, "
    "using the player number, the zone and one supporting number (point chance, block chance "
    "or the gap in metres). Sound energetic and natural, vary your phrasing, do not "
    "repeat the recent lines, and output only the sentence - no quotes, no preamble."
)

DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


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


class ClaudeBackend:
    name = "CLAUDE"

    def __init__(self, model: str = DEFAULT_CLAUDE_MODEL):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def generate(self, kind: str, facts: dict, recent: list[str]) -> str | None:
        anthropic = self._anthropic
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=2048,
                # Short single-sentence task: low effort keeps latency down.
                output_config={"effort": "low"},
                # Re-route to a fallback model if a safety classifier declines.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_prompt(kind, facts, recent)}],
            )
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError):
            raise
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APIStatusError) as exc:
            print(f"  [nlp] Claude request failed ({type(exc).__name__}); using template for this line")
            return None
        if response.stop_reason == "refusal":
            return None
        text = next((b.text for b in response.content if b.type == "text"), "")
        return _clean(text) or None


class LocalLLMBackend:
    name = "LOCAL LLM"

    def __init__(self, model: str = DEFAULT_LOCAL_MODEL):
        from transformers import pipeline

        print(f"  [nlp] loading local model {model} (first run downloads it)…")
        self.pipe = pipeline("text-generation", model=model, device=-1)

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


def build_backend(mode: str = "auto", claude_model: str = DEFAULT_CLAUDE_MODEL,
                  local_model: str = DEFAULT_LOCAL_MODEL):
    """Return a ready backend, or None for template-only commentary."""
    if mode == "template":
        return None

    if mode in ("auto", "claude"):
        try:
            backend = ClaudeBackend(claude_model)
            has_env_creds = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
            if mode == "claude" or has_env_creds:
                return backend
            # No env credentials: probe once so an `ant auth login` profile still works.
            backend.generate("insight", {"probe": True}, [])
            return backend
        except ImportError:
            reason = "anthropic SDK not installed (pip install anthropic)"
        except Exception as exc:  # missing/invalid credentials
            reason = f"{type(exc).__name__}: {exc}".splitlines()[0][:120]
        print(f"  [nlp] Claude unavailable - {reason}")
        if mode == "claude":
            return None

    if mode in ("auto", "local"):
        try:
            return LocalLLMBackend(local_model)
        except ImportError:
            print("  [nlp] local LLM unavailable - transformers not installed (pip install transformers)")
        except Exception as exc:
            print(f"  [nlp] local LLM unavailable - {type(exc).__name__}: {str(exc)[:120]}")
    return None
