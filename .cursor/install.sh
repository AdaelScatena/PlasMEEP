#!/usr/bin/env bash
#
# Idempotent environment bootstrap for PlasMEEP.
#
# PlasMEEP depends on Meep (pymeep/pymeep-extras), which is only distributed
# through conda-forge, so this script installs Miniforge (conda), creates a
# dedicated `plasmeep` environment with Meep + MPB, installs the Python
# dependencies, and installs PlasMEEP itself in editable mode.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MINIFORGE_DIR="${HOME}/miniforge3"
ENV_NAME="plasmeep"

# 1. Install Miniforge if conda is not already present.
if [ ! -x "${MINIFORGE_DIR}/bin/conda" ]; then
    echo "Installing Miniforge to ${MINIFORGE_DIR}..."
    curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash /tmp/miniforge.sh -b -p "${MINIFORGE_DIR}"
    rm -f /tmp/miniforge.sh
fi

# shellcheck disable=SC1091
source "${MINIFORGE_DIR}/etc/profile.d/conda.sh"

# 2. Create the conda environment with Meep if it does not exist yet.
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Creating conda environment '${ENV_NAME}' with pymeep + pymeep-extras..."
    conda create -n "${ENV_NAME}" -c conda-forge pymeep pymeep-extras -y
fi

conda activate "${ENV_NAME}"

# 3. Install the remaining Python dependencies and PlasMEEP (editable).
echo "Installing Python requirements..."
pip install -r "${REPO_ROOT}/requirements.txt"
pip install -e "${REPO_ROOT}"

# 4. Make conda + the plasmeep env available in interactive agent shells.
ACTIVATE_LINE="source ${MINIFORGE_DIR}/etc/profile.d/conda.sh && conda activate ${ENV_NAME}"
if ! grep -qF "${ACTIVATE_LINE}" "${HOME}/.bashrc" 2>/dev/null; then
    echo "${ACTIVATE_LINE}" >> "${HOME}/.bashrc"
fi

echo "PlasMEEP environment setup complete."
