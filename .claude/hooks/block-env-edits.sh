#!/usr/bin/env bash
# PreToolUse hook: block Edit/Write/MultiEdit on .env files.
# Reads the tool-call JSON from stdin. Exit 2 = block; exit 0 = allow.
set -u

path=$(jq -r '.tool_input.file_path // empty' 2>/dev/null)
if [[ -z "$path" ]]; then
  exit 0
fi

# Match .env at repo root or any subdirectory, but NOT .env.example / .env.local etc.
case "$(basename "$path")" in
  .env)
    echo ".env edits are blocked. Modify .env.example instead, or edit .env manually outside Claude." >&2
    exit 2
    ;;
esac
exit 0
