#!/usr/bin/env bash
# Clone the upstream simulators and build a working Python environment.
# Safe to re-run: every step is idempotent.
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p third_party results figures

clone() {  # clone <url> <dir>
    if [ -d "third_party/$2/.git" ]; then
        echo "  $2 already cloned"
    else
        echo "  cloning $2 ..."
        git clone --depth 1 "$1" "third_party/$2"
    fi
}

echo "== upstream simulators =="
# Digital PUM, from the DARTH-PUM authors' own lab (MIT).
# We read src/bottom/ReRAM_oscar.h for the OSCAR cost table. No build required.
clone https://github.com/ARCANA-Research/MASTODON.git MASTODON
# Analog crossbar accuracy simulator, Sandia National Labs (BSD-3).
# Used as a library; supplies the MILO ReRAM noise model as RRAMMilo.
clone https://github.com/sandialabs/cross-sim.git cross-sim

echo
echo "== python environment =="
if [ ! -d .venv ]; then
    python3 -m venv .venv
    echo "  created .venv"
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --quiet --upgrade pip
echo "  installing dependencies (this pulls PyTorch, ~800 MB, give it a few minutes)"
python -m pip install --quiet -r requirements.txt
# CrossSim is not on PyPI -- it must be installed from the clone.
python -m pip install --quiet -e third_party/cross-sim

echo
echo "== verifying =="
python -c "
import numpy, torch, matplotlib
print(f'  numpy       {numpy.__version__}')
print(f'  torch       {torch.__version__}  (cuda: {torch.cuda.is_available()})')
print(f'  matplotlib  {matplotlib.__version__}')
import simulator
print('  crosssim    ok')
from simulator.devices.custom.RRAM_Milo import RRAMMilo
print('  RRAMMilo    ok  (the MILO programming-noise model the paper uses)')
"
python -m darthpum.recipes >/dev/null && echo "  OSCAR recipes parsed ok"
python -m darthpum.area    >/dev/null && echo "  area model ok"

echo
echo "Done. Activate with:  source .venv/bin/activate"
