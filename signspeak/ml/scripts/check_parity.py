"""Assert that the Python and TypeScript feature encoders agree.

`ml/signspeak_ml/features.py` runs at training time; `apps/web/lib/features.ts`
runs in the browser at inference time. If they drift by even a little, the
model receives inputs from a distribution it never saw, accuracy collapses, and
nothing anywhere reports an error. You find out weeks later, from a user, as
"it works worse than the notebook said".

This script generates fixture landmark inputs, encodes them in Python, encodes
the same inputs in TypeScript via node, and compares element by element.

    python scripts/check_parity.py            # needs node + typescript

Put it in CI. If node or typescript is unavailable it skips rather than fails,
so it never blocks a machine that only does training — but a skip is not a
pass, and the message says so.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from signspeak_ml.features import STATIC_DIM, encode_frame  # noqa: E402

TOL = 1e-4


def fixture(seed: int) -> dict:
    """Landmark dicts in the exact shape MediaPipe emits."""
    rng = np.random.default_rng(seed)

    def pose():
        cx, cy, s = rng.uniform(0.35, 0.65), rng.uniform(0.35, 0.6), rng.uniform(0.6, 1.3)
        p = np.zeros((33, 3))
        p[0] = [cx, cy - 0.22 * s, 0]
        p[2] = [cx - 0.03 * s, cy - 0.24 * s, 0]
        p[5] = [cx + 0.03 * s, cy - 0.24 * s, 0]
        p[11] = [cx + 0.12 * s, cy, 0.01]
        p[12] = [cx - 0.12 * s, cy, -0.01]
        p[13] = [cx + 0.18 * s, cy + 0.15 * s, 0]
        p[14] = [cx - 0.18 * s, cy + 0.15 * s, 0]
        p[15] = [cx + 0.20 * s, cy + 0.28 * s, 0]
        p[16] = [cx - 0.20 * s, cy + 0.28 * s, 0]
        p[23] = [cx + 0.09 * s, cy + 0.34 * s, 0]
        p[24] = [cx - 0.09 * s, cy + 0.34 * s, 0]
        return p + rng.normal(0, 0.004, (33, 3))

    def hand():
        c = rng.uniform(0.3, 0.7, 2)
        return np.stack(
            [[c[0] + 0.05 * np.cos(i * 0.31), c[1] + 0.05 * np.sin(i * 0.31), 0.008 * i]
             for i in range(21)]
        ) + rng.normal(0, 0.002, (21, 3))

    drop = rng.random()
    return {
        "pose": pose(),
        # Cover the dropped-hand path too: that is where a missing presence
        # flag or a wrong offset shows up.
        "left_hand": None if drop < 0.2 else hand(),
        "right_hand": None if 0.8 < drop else hand(),
        "face": None,
        "dominant": "left" if seed % 4 == 3 else "right",
    }


HARNESS = r"""
import ts from "TSLIB";
import fs from "fs";
import path from "path";
import { pathToFileURL } from "url";

const featuresPath = process.argv[2];
const fixturePath  = process.argv[3];
const outPath      = process.argv[4];

// Transpile features.ts to ESM JS in place, then import it.
const src = fs.readFileSync(featuresPath, "utf8");
const js = ts.transpileModule(src, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
}).outputText;
const tmp = path.join(path.dirname(outPath), "features.mjs");
fs.writeFileSync(tmp, js);
const mod = await import(pathToFileURL(tmp).href);

const toLm = (a) => a === null ? null : a.map(([x, y, z]) => ({ x, y, z }));
const cases = JSON.parse(fs.readFileSync(fixturePath, "utf8"));
const out = cases.map((c) => {
  const v = mod.encodeFrame(
    { pose: toLm(c.pose), leftHand: toLm(c.left_hand), rightHand: toLm(c.right_hand), face: null },
    c.dominant,
  );
  return v === null ? null : Array.from(v);
});
fs.writeFileSync(outPath, JSON.stringify(out));
"""


def find_typescript() -> str | None:
    for base in (REPO / "apps/web/node_modules", REPO / "node_modules", Path("/tmp/claude-0/tsc/node_modules")):
        p = base / "typescript/lib/typescript.js"
        if p.exists():
            return str(p)
    return None


def main() -> int:
    node = shutil.which("node")
    tslib = find_typescript()
    if not node or not tslib:
        print("SKIP: needs node and typescript (run `npm install` in apps/web).")
        print("      A skip is not a pass — the two encoders are unverified.")
        return 0

    cases = [fixture(s) for s in range(24)]
    expected = [encode_frame(c["pose"], c["left_hand"], c["right_hand"], None, c["dominant"])
                for c in cases]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "fixtures.json").write_text(json.dumps([
            {
                "pose": c["pose"].tolist(),
                "left_hand": None if c["left_hand"] is None else c["left_hand"].tolist(),
                "right_hand": None if c["right_hand"] is None else c["right_hand"].tolist(),
                "dominant": c["dominant"],
            }
            for c in cases
        ]))
        (tmp / "harness.mjs").write_text(HARNESS.replace("TSLIB", tslib))

        r = subprocess.run(
            [node, str(tmp / "harness.mjs"),
             str(REPO / "apps/web/lib/features.ts"),
             str(tmp / "fixtures.json"), str(tmp / "out.json")],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print("FAIL: the TypeScript harness did not run.")
            print(r.stderr[-2500:])
            return 1

        got = json.loads((tmp / "out.json").read_text())

    fails = 0
    worst = 0.0
    for i, (py, tsv) in enumerate(zip(expected, got)):
        if (py is None) != (tsv is None):
            print(f"FAIL case {i}: one encoder returned None and the other did not "
                  f"(py={py is None}, ts={tsv is None})")
            fails += 1
            continue
        if py is None:
            continue
        if len(tsv) != STATIC_DIM:
            print(f"FAIL case {i}: TS returned {len(tsv)} floats, Python expects {STATIC_DIM}")
            fails += 1
            continue
        d = float(np.abs(np.asarray(tsv, dtype=np.float64) - py.astype(np.float64)).max())
        worst = max(worst, d)
        if d > TOL:
            bad = int(np.argmax(np.abs(np.asarray(tsv) - py)))
            print(f"FAIL case {i}: max |Δ| = {d:.2e} at index {bad} "
                  f"(py={py[bad]:.6f}, ts={tsv[bad]:.6f})")
            fails += 1

    if fails:
        print(f"\n{fails}/{len(cases)} cases diverge. The browser is running a DIFFERENT "
              f"feature encoder from the trainer — fix before trusting any metric.")
        return 1

    print(f"PARITY OK — {len(cases)} cases, max |Δ| = {worst:.2e} (tolerance {TOL:.0e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
