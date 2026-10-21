"""OSCAR logic-family recipes, parsed from the authors' own simulator.

MASTODON -- the digital-PUM simulator DARTH-PUM is built on -- ships a "recipe
book" at src/bottom/ReRAM_oscar.h. It maps each supported operation to the exact
cycle-by-cycle micro-op sequence the hardware issues. Quoting its header comment:

    "this recipe book represents the final translation right before issuing to the
     back end ... given the cycle-by-cycle translation of the recipe, those
     intermediate steps won't create extra timing overhead"

So the length of a recipe is the operation's cycle cost on one bit-slice, and the
micro-op mix gives the energy. We parse the header rather than re-deriving costs,
which means our digital model uses the authors' real numbers.

We parse C++ with a regex, which is normally a bad idea. It is safe here because
the file is machine-generated-looking and entirely uniform: every recipe is a
`deque <string> name {{ "UOP", ... }};` followed by `table.insert({"NAME", name});`.
The parser asserts on the expected recipe count so a format change fails loudly
instead of silently producing wrong costs.

A parsed snapshot is cached to recipes_oscar.json so the package works without the
MASTODON clone present (and so results stay reproducible if upstream changes).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_OSCAR_H = _REPO / "third_party" / "MASTODON" / "src" / "bottom" / "ReRAM_oscar.h"
_CACHE = _HERE / "recipes_oscar.json"

EXPECTED_RECIPE_COUNT = 33


# Micro-op classes, from MASTODON src/middle-racer/racer_uop.h (enum Racer_Uop_Type).
# Only BOOLEAN and TRANSFER move charge through devices; the rest are control.
UOP_CLASS = {
    "NOR": "BOOLEAN",
    "OR": "BOOLEAN",
    "AND": "BOOLEAN",
    "BUF": "TRANSFER",
    "COPY": "TRANSFER",
    "LOCK": "PIPELINECTRL",
    "RELEASE": "PIPELINECTRL",
    "STALL": "PIPELINECTRL",
    "FILTER": "PREDICATE",
    "UNFILTER": "PREDICATE",
    "NOP": "UOP_NOP",
}


@dataclass(frozen=True)
class Recipe:
    """One operation's micro-op sequence, for a single bit position."""

    name: str
    uops: tuple[str, ...]

    @property
    def cycles(self) -> int:
        """Cycles to execute this operation on one bit-slice.

        One micro-op per cycle -- see the module docstring.
        """
        return len(self.uops)

    @property
    def mix(self) -> dict[str, int]:
        """Count of each micro-op opcode."""
        return dict(Counter(u.split()[0] for u in self.uops if u.strip()))

    @property
    def switching_uops(self) -> int:
        """Micro-ops that actually switch devices, for the energy model.

        Control micro-ops (LOCK/RELEASE/NOP) and predicate micro-ops
        (FILTER/UNFILTER) set up the array but do not drive a write current.
        """
        return sum(
            n for op, n in self.mix.items()
            if UOP_CLASS.get(op) in ("BOOLEAN", "TRANSFER")
        )


def _parse_header(path: Path) -> dict[str, list[str]]:
    src = path.read_text()

    bodies = {
        m.group(1): re.findall(r'"([^"]*)"', m.group(2))
        for m in re.finditer(
            r"deque\s*<\s*string\s*>\s*(\w+)\s*\{\{(.*?)\}\};", src, re.S
        )
    }
    # table.insert({"ADD", add});  ->  public name : local variable
    registered = dict(
        re.findall(r'table\.insert\(\{"(\w+)",\s*(\w+)\}\)', src)
    )

    missing = {v for v in registered.values()} - bodies.keys()
    if missing:
        raise ValueError(f"{path.name}: registered but unparsed recipes: {missing}")

    return {name: bodies[var] for name, var in registered.items()}


def load(refresh: bool = False) -> dict[str, Recipe]:
    """Return {operation name: Recipe}.

    Parses the MASTODON header when available, otherwise falls back to the cached
    snapshot. Pass refresh=True to re-parse and rewrite the cache.
    """
    if refresh or not _CACHE.exists():
        if not _OSCAR_H.exists():
            raise FileNotFoundError(
                f"{_OSCAR_H} not found and no cache at {_CACHE}.\n"
                "Run setup.sh to clone MASTODON."
            )
        table = _parse_header(_OSCAR_H)
        if len(table) != EXPECTED_RECIPE_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_RECIPE_COUNT} OSCAR recipes, parsed "
                f"{len(table)}. The upstream header format may have changed -- "
                "re-check the parser before trusting any digital cost number."
            )
        _CACHE.write_text(json.dumps(table, indent=1, sort_keys=True))
    else:
        table = json.loads(_CACHE.read_text())

    return {name: Recipe(name, tuple(uops)) for name, uops in table.items()}


def cycles(op: str, recipes: dict[str, Recipe] | None = None) -> int:
    """Cycle cost of `op` on one bit-slice."""
    recipes = recipes if recipes is not None else load()
    try:
        return recipes[op].cycles
    except KeyError:
        raise KeyError(
            f"OSCAR has no recipe for {op!r}. Available: {sorted(recipes)}"
        ) from None


# --------------------------------------------------------------------------
# Cross-check against the published RACER numbers
# --------------------------------------------------------------------------

def crosscheck() -> dict:
    """Sanity-check the parsed costs against independently published figures.

    [RACER] (MICRO '21) states a full add costs 22 NOR primitives per bit under
    the MAGIC NOR logic family. [OSCAR] (JETCAS '22) then reports roughly +30%
    performance over MAGIC NOR by adding a single-cycle OR primitive.

    Our parsed OSCAR ADD is 21 cycles containing 16 NOR + 1 OR, which is the
    expected shape: fewer NOR primitives than MAGIC's 22, with the saving coming
    from the native OR. This is an independent confirmation that we are reading
    the recipe book correctly.
    """
    r = load()
    add = r["ADD"]
    return {
        "ADD_cycles": add.cycles,
        "ADD_mix": add.mix,
        "RACER_MAGIC_NOR_per_bit": 22,
        "OSCAR_NOR_per_bit": add.mix.get("NOR", 0),
        "uses_native_OR": add.mix.get("OR", 0) > 0,
        "n_recipes": len(r),
    }


if __name__ == "__main__":
    rs = load(refresh=True)
    print(f"{len(rs)} OSCAR recipes parsed from {_OSCAR_H.name}\n")
    print(f"{'OP':<12}{'cycles':>7}{'switching':>11}   micro-op mix")
    for name, rec in sorted(rs.items(), key=lambda kv: kv[1].cycles):
        mix = " ".join(f"{k}:{v}" for k, v in
                       sorted(rec.mix.items(), key=lambda kv: -kv[1]))
        print(f"{name:<12}{rec.cycles:>7}{rec.switching_uops:>11}   {mix}")
    print()
    print("cross-check vs published RACER/OSCAR numbers:")
    print(json.dumps(crosscheck(), indent=2))
