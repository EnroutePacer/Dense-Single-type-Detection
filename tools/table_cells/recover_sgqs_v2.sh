#!/usr/bin/env bash
# Compatibility convenience; topology, output and attempt are explicit CLI arguments.
set -Eeuo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_sgqs_v2.sh" --mode recover "$@"
