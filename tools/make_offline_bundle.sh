#!/usr/bin/env bash
# Self-contained source archive for servers without GitHub access:
# pydbsr + O. Zatsarinny's DBSR3, LIBRARIES, UTILS at the pinned commits (upstream/).
#   bash tools/make_offline_bundle.sh [out.tar.gz]
# On the server:  tar xzf pydbsr_offline.tar.gz && cd pydbsr_src && pip install --no-build-isolation -v .
set -euo pipefail
OUT="$(realpath -m "${1:-pydbsr_offline.tar.gz}")"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/pydbsr_src/upstream"
git -C "$REPO" archive HEAD | tar -x -C "$TMP/pydbsr_src"
for r in DBSR3 LIBRARIES UTILS; do
  sha=$(sed -n "s/^set(DBSR_COMMIT_$r *\([0-9a-f]*\))/\1/p" "$REPO/cmake/FetchUpstream.cmake")
  git clone -q "https://github.com/zatsaroi/$r.git" "$TMP/$r"
  git -C "$TMP/$r" checkout -q "$sha"
  mkdir -p "$TMP/pydbsr_src/upstream/$r"
  git -C "$TMP/$r" archive HEAD | tar -x -C "$TMP/pydbsr_src/upstream/$r"
done
tar czf "$OUT" -C "$TMP" pydbsr_src
echo "written $OUT"
