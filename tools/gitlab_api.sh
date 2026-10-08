#!/usr/bin/env bash
# GitLab REST API of one project only: alex.undr.a17/spectroscopy_lib (issues, comments).
#   bash tools/gitlab_api.sh GET  issues
#   bash tools/gitlab_api.sh POST issues body.json
#   bash tools/gitlab_api.sh POST issues/3/notes body.json
# Token: $GITLAB_TOCKEN, a project access token set in the cloud environment settings.
# Allowed in .claude/settings.json; it cannot reach other projects and has no DELETE.
set -euo pipefail
method=${1:?usage: gitlab_api.sh GET|POST|PUT path [body.json]}
path=${2:?usage: gitlab_api.sh GET|POST|PUT path [body.json]}
body=${3:-}
case "$method" in GET|POST|PUT) ;; *) echo "method must be GET, POST or PUT" >&2; exit 2 ;; esac
case "$path" in *..*|/*|*://*) echo "path must be relative to the project" >&2; exit 2 ;; esac
url="https://gitlab.com/api/v4/projects/alex.undr.a17%2Fspectroscopy_lib/$path"
args=(-sS -X "$method" -H "PRIVATE-TOKEN: ${GITLAB_TOCKEN:?GITLAB_TOCKEN is not set}")
if [ -n "$body" ]; then
  args+=(-H "Content-Type: application/json" --data "@$body")
fi
curl "${args[@]}" "$url"
