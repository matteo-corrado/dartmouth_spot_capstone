"""Generate the LLM grammar and the prompt's action list from the core intents
and the registered features (Task 16). Pure: pass a Registry.

Only actions with a prompt_line reach the LLM. walk_to (regex-only, with
unclamped relative x/y/yaw) has none, so the model can never emit it (R23).
"""
from __future__ import annotations

from pathlib import Path

from src.spot.core.core_intents import all_actions

TEMPLATE_PATH = Path(__file__).with_name("spot_action.gbnf.tmpl")
# One generic params rule pair; per-action grammar_params rules follow it.
_GENERIC_PARAMS = ('params ::= "{" ws ( param ( "," ws param )* )? ws "}"\n'
                   'param ::= string ":" ws value')


def llm_actions(registry, key: str, exclude=frozenset()):
    """Actions the LLM may emit, in `key` order (grammar_rank / prompt_rank)."""
    acts = [a for a in all_actions(registry) if a.prompt_line and a.name not in exclude]
    return sorted(acts, key=lambda a: getattr(a, key))  # stable: ties keep core-then-load order


def build_grammar(registry) -> str:
    acts = llm_actions(registry, "grammar_rank")
    # one line: llama.cpp's GBNF parser fails silently on multi-line `|`
    names = "action-name ::= " + " | ".join(f'"\\"{a.name}\\""' for a in acts)
    params = "\n".join([_GENERIC_PARAMS] + [a.grammar_params for a in acts if a.grammar_params])
    text = TEMPLATE_PATH.read_text(encoding="utf-8")
    return text.replace("{{ACTION_NAMES}}", names).replace("{{PARAMS}}", params)


def build_prompt_actions(registry, exclude=frozenset()) -> str:
    """The AVAILABLE ACTIONS bullets: one per line (a line that already ends
    in a newline, like emote's, is the separator itself), no trailing newline."""
    lines = [a.prompt_line for a in llm_actions(registry, "prompt_rank", exclude)]
    text = "".join(l if l.endswith("\n") else l + "\n" for l in lines)
    return text[:-1] if text.endswith("\n") else text
