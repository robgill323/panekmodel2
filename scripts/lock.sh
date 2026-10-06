#!/usr/bin/env bash
#
# Regenerate the hash-pinned dependency locks with pip-tools.
#
#   ./scripts/lock.sh              # locks for THIS platform (the verified path)
#   ./scripts/lock.sh --docker     # lock for the Linux image (UNVERIFIED)
#
# Input: pyproject.toml. pip-tools reads the [project] dependencies and the
# [project.optional-dependencies] dev extra directly, so pyproject.toml plays
# the role a requirements.in plays elsewhere. A separate requirements.in would
# be a second list of the same dependencies, and two lists drift.
#
# Output:
#   requirements.lock       runtime dependencies, exact pins + sha256 hashes
#   requirements-dev.lock   runtime + the dev extra (pytest, httpx, pyyaml)
#
# A LOCK IS ONLY VALID FOR THE PLATFORM AND PYTHON VERSION THAT PRODUCED IT.
# pip-tools resolves for the running interpreter and drops dependencies whose
# environment markers do not apply — a macOS lock lacks the Linux-only
# nvidia-* and triton wheels that torch and whisper declare there. Installing
# it on Linux under --require-hashes fails loudly rather than silently
# installing something unpinned, which is the right failure, but it means the
# committed locks are not the Docker image's lock. Hence --docker.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

# The header pip-compile writes would otherwise echo a command line with
# artefacts such as `--no-index --index-url=None` that are not what ran.
# (pip-tools 7.4.1 takes this from the environment, not a flag.)
export CUSTOM_COMPILE_COMMAND="./scripts/lock.sh  (input: pyproject.toml)"
COMPILE_FLAGS=(--generate-hashes --allow-unsafe --strip-extras --quiet)

# The version this was verified with. pip-tools 7.6.1 builds the PyPI JSON-API
# URL wrongly (https://pypi.org/<name>/json, a 404), silently falls back to
# downloading EVERY wheel of every platform to hash it, and ran past 22 GB of
# torch wheels before it was stopped. 7.4.1 reads the hashes from the JSON API
# and locks this project in about a minute. Override at your own risk.
PIP_TOOLS_VERSION="7.4.1"
# pip-tools drives pip's resolver and finder internals, so the PAIR is what was
# verified, not pip-tools alone.
PIP_VERSION="24.2"

have_pip_tools() {
    "$1" -c 'import piptools' >/dev/null 2>&1
}

check_pip_tools_version() {
    local tools pip
    tools="$("$PYTHON" -c 'from importlib.metadata import version; print(version("pip-tools"))')"
    pip="$("$PYTHON" -c 'from importlib.metadata import version; print(version("pip"))')"
    if { [ "$tools" != "$PIP_TOOLS_VERSION" ] || [ "$pip" != "$PIP_VERSION" ]; } \
            && [ -z "${LOCK_ANY_PIP_TOOLS:-}" ]; then
        echo "found pip-tools $tools with pip $pip; this script is verified with" >&2
        echo "pip-tools $PIP_TOOLS_VERSION + pip $PIP_VERSION:" >&2
        echo "  $PYTHON -m pip install 'pip==$PIP_VERSION' 'pip-tools==$PIP_TOOLS_VERSION'" >&2
        echo "or set LOCK_ANY_PIP_TOOLS=1 to try them anyway (see the note above)." >&2
        exit 1
    fi
}

lock_local() {
    if ! have_pip_tools "$PYTHON"; then
        echo "pip-tools is not installed for $PYTHON." >&2
        echo "Install it into a throwaway environment, not the project venv:" >&2
        echo "  python3.12 -m venv /tmp/lockenv" >&2
        echo "  /tmp/lockenv/bin/pip install 'pip==$PIP_VERSION' 'pip-tools==$PIP_TOOLS_VERSION'" >&2
        echo "  PYTHON=/tmp/lockenv/bin/python ./scripts/lock.sh" >&2
        exit 1
    fi
    check_pip_tools_version

    echo "==> requirements.lock (runtime)"
    "$PYTHON" -m piptools compile pyproject.toml "${COMPILE_FLAGS[@]}" \
        --output-file requirements.lock

    echo "==> requirements-dev.lock (runtime + dev extra)"
    "$PYTHON" -m piptools compile pyproject.toml --extra dev "${COMPILE_FLAGS[@]}" \
        --output-file requirements-dev.lock

    cat <<'EOF'

Locks written. Verify them in a clean environment before trusting them —
a lock that has never been installed from is a guess:

  python3.12 -m venv /tmp/lockcheck
  /tmp/lockcheck/bin/pip install --require-hashes -r requirements-dev.lock
  /tmp/lockcheck/bin/pip install --no-deps --no-build-isolation -e .
  /tmp/lockcheck/bin/python -m pytest -m "not slow"
EOF
}

lock_docker() {
    cat <<'EOF' >&2
WARNING: this path has never been executed. It was written on a machine with
no running Docker daemon. Read the command below before running it, and check
the resulting lock pins the torch build you expect.

EOF
    if ! docker info >/dev/null 2>&1; then
        echo "The Docker daemon is not reachable; start it and retry." >&2
        exit 1
    fi

    # Resolved INSIDE the base image the project runs in, so the platform and
    # Python version match the target rather than this host. The torch index
    # is passed through so the lock pins the same build the Dockerfile
    # installs — otherwise it would name the PyPI (CUDA) wheel.
    docker run --rm \
        -v "$PWD:/work" -w /work \
        -e TORCH_INDEX_URL="$TORCH_INDEX_URL" \
        python:3.12-slim \
        sh -euc '
            pip install --quiet --root-user-action=ignore "pip==24.2" "pip-tools==7.4.1"
            pip-compile pyproject.toml \
                --generate-hashes --allow-unsafe --strip-extras \
                --index-url "$TORCH_INDEX_URL" \
                --extra-index-url https://pypi.org/simple \
                --output-file requirements-docker.lock
        '

    cat <<'EOF'

requirements-docker.lock written. Build against it, then confirm the
container agrees about its own torch build:

  REQUIREMENTS_FILE=requirements-docker.lock docker compose build
  docker compose run --rm --entrypoint python throughline \
      -c "import torch; print(torch.__version__, torch.cuda.is_available())"
EOF
}

case "${1:-}" in
    "")        lock_local ;;
    --docker)  lock_docker ;;
    *)         echo "usage: $0 [--docker]" >&2; exit 2 ;;
esac
