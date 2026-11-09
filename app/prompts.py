"""Load, validate and render versioned prompts from `prompts/triage/*.yaml`.

* Every prompt file is validated at startup (required keys, known model alias,
  template placeholders are plain identifiers).
* Rendering uses `str.format` over validated placeholders only. Values (such as
  the ticket) are never re-parsed, so braces in user text are harmless.
* The ticket is untrusted: any `<ticket>` / `</ticket>` tags inside it are
  neutralised so it cannot break out of the delimiter in the prompt.
* The active prompt per role comes from settings; an optional canary splits
  traffic between versions, deterministically per request key.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any, Mapping

import yaml

from app.schemas import Category, Intent

VALID_ALIASES = frozenset({"primary", "strong", "small"})
REQUIRED_KEYS = ("id", "model_alias", "params", "system", "template")
BUILTIN_VARS = frozenset({"categories", "intents", "few_shot_examples"})

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TICKET_TAG = re.compile(r"<\s*/?\s*ticket\s*>", re.IGNORECASE)


class PromptError(ValueError):
    """Raised for invalid prompt files, bad config, or rendering problems."""


def sanitize_ticket(text: Any) -> str:
    """Neutralise <ticket> delimiter tags in untrusted text."""
    return _TICKET_TAG.sub("[removed-tag]", str(text))


def _placeholders(template: str, where: str) -> set[str]:
    """Return placeholder names in `template`; reject anything but plain `{name}`."""
    names: set[str] = set()
    try:
        parsed = list(Formatter().parse(template))
    except ValueError as exc:
        raise PromptError(f"{where}: malformed template braces: {exc}") from exc
    for _, name, spec, conv in parsed:
        if name is None:
            continue
        if not _IDENT.match(name) or spec or conv:
            raise PromptError(f"{where}: unsupported placeholder '{{{name}}}'")
        names.add(name)
    return names


@dataclass(frozen=True)
class RenderedPrompt:
    """A rendered prompt ready for the gateway."""

    messages: list[dict[str, str]]
    prompt_id: str
    model_alias: str
    params: dict[str, Any]


@dataclass(frozen=True)
class Prompt:
    """One validated prompt definition."""

    id: str
    model_alias: str
    params: dict[str, Any]
    system: str
    template: str
    few_shot: tuple[dict[str, Any], ...] = ()
    source: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], source: str = "<dict>") -> "Prompt":
        """Validate raw YAML data and build a Prompt."""
        if not isinstance(data, Mapping):
            raise PromptError(f"{source}: top level must be a mapping")
        missing = [k for k in REQUIRED_KEYS if k not in data]
        if missing:
            raise PromptError(f"{source}: missing required keys {missing}")
        pid = data["id"]
        if not isinstance(pid, str) or not pid.strip():
            raise PromptError(f"{source}: 'id' must be a non-empty string")
        if data["model_alias"] not in VALID_ALIASES:
            raise PromptError(
                f"{source}: model_alias '{data['model_alias']}' not in {sorted(VALID_ALIASES)}"
            )
        params = data["params"]
        if not isinstance(params, Mapping):
            raise PromptError(f"{source}: 'params' must be a mapping")
        if not isinstance(params.get("temperature"), (int, float)) or isinstance(
            params.get("temperature"), bool
        ):
            raise PromptError(f"{source}: params.temperature must be a number")
        max_tokens = params.get("max_tokens")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise PromptError(f"{source}: params.max_tokens must be a positive integer")
        for key in ("system", "template"):
            if not isinstance(data[key], str) or not data[key].strip():
                raise PromptError(f"{source}: '{key}' must be a non-empty string")
        few_shot = data.get("few_shot") or []
        if not isinstance(few_shot, list) or not all(
            isinstance(e, Mapping) and "ticket" in e and "output" in e for e in few_shot
        ):
            raise PromptError(f"{source}: 'few_shot' must be a list of {{ticket, output}} items")

        prompt = cls(
            id=pid.strip(),
            model_alias=data["model_alias"],
            params=dict(params),
            system=data["system"],
            template=data["template"],
            few_shot=tuple(dict(e) for e in few_shot),
            source=source,
        )
        prompt.placeholders  # validates both templates eagerly
        return prompt

    @property
    def placeholders(self) -> set[str]:
        """All placeholder names used by the system and user templates."""
        return _placeholders(self.system, f"{self.source} system") | _placeholders(
            self.template, f"{self.source} template"
        )

    @property
    def required_vars(self) -> set[str]:
        """Variables the caller must supply (everything except built-ins)."""
        return self.placeholders - BUILTIN_VARS

    def _few_shot_block(self) -> str:
        if not self.few_shot:
            return ""
        parts = [
            f"<ticket>\n{sanitize_ticket(ex['ticket'])}\n</ticket>\n"
            f"{json.dumps(ex['output'], ensure_ascii=False)}"
            for ex in self.few_shot
        ]
        return "Examples:\n\n" + "\n\n".join(parts)

    def render(self, **variables: Any) -> RenderedPrompt:
        """Render system + user messages.

        Raises:
            PromptError: if a required variable is missing.
        """
        values: dict[str, Any] = {
            "categories": ", ".join(c.value for c in Category),
            "intents": ", ".join(i.value for i in Intent),
            "few_shot_examples": self._few_shot_block(),
        }
        values.update(variables)
        if "ticket" in values:
            values["ticket"] = sanitize_ticket(values["ticket"])
        missing = sorted(self.placeholders - values.keys())
        if missing:
            raise PromptError(f"{self.id}: missing variables {missing}")
        return RenderedPrompt(
            messages=[
                {"role": "system", "content": self.system.format(**values).strip()},
                {"role": "user", "content": self.template.format(**values).strip()},
            ],
            prompt_id=self.id,
            model_alias=self.model_alias,
            params=dict(self.params),
        )


class PromptRegistry:
    """All loaded prompts plus active-version and canary selection.

    `canary` maps role -> {prompt_id: weight} and describes the FULL traffic split
    for that role (include the control version too). Roles without a canary use
    the id in `active`.
    """

    def __init__(
        self,
        prompts_dir: Path,
        active: Mapping[str, str],
        canary: Mapping[str, Mapping[str, float]] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._prompts: dict[str, Prompt] = {}
        self._active = dict(active)
        self._canary = {r: dict(v) for r, v in (canary or {}).items()}
        self._rng = rng or random.Random()
        self._load(Path(prompts_dir))
        self._validate_config()

    @classmethod
    def from_settings(cls, settings: Any) -> "PromptRegistry":
        """Build a registry from `app.config.Settings`."""
        return cls(settings.prompts_dir, settings.active_prompts, settings.prompt_canary)

    # -- loading ------------------------------------------------------------
    def _load(self, directory: Path) -> None:
        files = sorted(directory.glob("*.yaml"))
        if not files:
            raise PromptError(f"No prompt YAML files found in {directory}")
        for path in files:
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                raise PromptError(f"{path.name}: invalid YAML: {exc}") from exc
            prompt = Prompt.from_dict(data, source=path.name)
            if prompt.id in self._prompts:
                raise PromptError(
                    f"Duplicate prompt id '{prompt.id}' in {path.name} and "
                    f"{self._prompts[prompt.id].source}"
                )
            self._prompts[prompt.id] = prompt

    def _validate_config(self) -> None:
        for role, pid in self._active.items():
            if pid not in self._prompts:
                raise PromptError(f"Active prompt for role '{role}' is unknown: '{pid}'")
        for role, variants in self._canary.items():
            if not variants:
                raise PromptError(f"Canary for role '{role}' is empty")
            for pid, weight in variants.items():
                if pid not in self._prompts:
                    raise PromptError(f"Canary for role '{role}' names unknown prompt '{pid}'")
                if not isinstance(weight, (int, float)) or weight < 0:
                    raise PromptError(f"Canary weight for '{pid}' must be >= 0")
            if sum(variants.values()) <= 0:
                raise PromptError(f"Canary weights for role '{role}' sum to zero")

    # -- access -------------------------------------------------------------
    @property
    def ids(self) -> list[str]:
        """All loaded prompt ids."""
        return sorted(self._prompts)

    def get(self, prompt_id: str) -> Prompt:
        """Return a prompt by id."""
        try:
            return self._prompts[prompt_id]
        except KeyError:
            raise PromptError(f"Unknown prompt id '{prompt_id}'") from None

    def select(self, role: str, key: str | None = None) -> str:
        """Choose the prompt id for a role.

        With a canary, the same `key` (e.g. client id) always lands on the same
        variant; with no key the choice is random.
        """
        if role not in self._active:
            raise PromptError(f"Unknown prompt role '{role}'")
        variants = self._canary.get(role)
        if not variants:
            return self._active[role]
        total = float(sum(variants.values()))
        if key is None:
            point = self._rng.random()
        else:
            digest = hashlib.sha256(f"{role}:{key}".encode()).hexdigest()
            point = int(digest[:8], 16) / 16**8
        cumulative = 0.0
        chosen = None
        for pid, weight in sorted(variants.items()):
            if weight <= 0:
                continue
            chosen = pid
            cumulative += weight / total
            if point < cumulative:
                return pid
        return chosen  # type: ignore[return-value]  # float rounding: last positive variant

    def render(self, prompt_id: str, **variables: Any) -> RenderedPrompt:
        """Render a specific prompt id."""
        return self.get(prompt_id).render(**variables)

    def render_for(self, role: str, key: str | None = None, **variables: Any) -> RenderedPrompt:
        """Select the prompt for `role` (honouring the canary) and render it."""
        return self.render(self.select(role, key), **variables)