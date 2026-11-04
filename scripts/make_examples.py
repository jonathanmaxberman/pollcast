"""Generate the starter configs shipped with the package from configs/reference.yaml.

Usage: uv run python scripts/make_examples.py

Writes src/pollcast/examples/{polls_only,polls_fundamentals,reference}.yaml.
They differ only in which adjustments are enabled, so every other
assumption (sources, error sizes, averaging) stays in step with the
reference configuration. Rerun after changing configs/reference.yaml.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs" / "reference.yaml"
OUT = ROOT / "src" / "pollcast" / "examples"

VARIANTS = {
    "polls_only": ("polls-only", "A plain polls-only forecast: time-weighted averages and the simulation, no adjustments.", set()),
    "polls_fundamentals": ("polls-fundamentals", "Polls blended with a fundamentals estimate (state lean, incumbency, national mood), weighted by lead time.",
                           {"senate_fundamentals_by_lead"}),
    "reference": ("reference", "The reference pollcast forecast: the combination of adjustments that scored best out of sample.", None),
}


def set_enabled(text: str, keep: set[str] | None) -> str:
    """Turn every indicator off except `keep` (None leaves the file as is)."""
    if keep is None:
        return text
    start = text.index("\nindicators:")
    end = text.index("\n", text.index("\nsimulate:") - 1) if "\nsimulate:" in text else len(text)
    head, block, tail = text[:start], text[start:end], text[end:]

    def flip(m: re.Match) -> str:
        name = m.group(1)
        value = "true" if name in keep else "false"
        return f"{m.group(0).split('enabled:')[0]}enabled: {value}"

    block = re.sub(r"\n  ([a-z_]+):\n(?:    (?!enabled:).*\n)*?    enabled: [^\n]*", flip, block)
    return head + block + tail


def main() -> None:
    text = SOURCE.read_text()
    OUT.mkdir(exist_ok=True)
    (OUT / "__init__.py").write_text('"""Starter configs written by `pollcast init`."""\n')
    for stem, (name, blurb, keep) in VARIANTS.items():
        body = set_enabled(text, keep)
        body = re.sub(r"^name: .*$", f"name: {name}", body, count=1, flags=re.M)
        body = re.sub(r"\A(#.*\n)+", "", body)
        header = (f"# {blurb}\n# Generated from configs/reference.yaml by scripts/make_examples.py.\n"
                  "# Every assumption lives here; edit freely. Turn an adjustment on only after\n"
                  "# `pollcast backtest` shows it passing.\n")
        (OUT / f"{stem}.yaml").write_text(header + body)
        print(f"wrote {OUT / (stem + '.yaml')}")


if __name__ == "__main__":
    main()
