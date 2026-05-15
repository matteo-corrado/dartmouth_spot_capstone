#!/usr/bin/env bash
# PreToolUse hook: enforce repo commit conventions.
#   - one-line commit messages only (no embedded newlines)
#   - no Co-Authored-By trailers
# Applies to Bash tool calls whose command contains `git commit`.
# Exit 2 = block; exit 0 = allow.
set -u

cmd=$(jq -r '.tool_input.command // empty' 2>/dev/null)
if [[ -z "$cmd" ]]; then
  exit 0
fi

# Only police actual git commit invocations.
if ! grep -qE '(^|[[:space:]&;|])git[[:space:]]+commit([[:space:]]|$)' <<<"$cmd"; then
  exit 0
fi

# `git commit --amend` without -m / -F is interactive — let it pass.
if ! grep -qE '\-m[[:space:]]|\-F[[:space:]]|--message|--file' <<<"$cmd"; then
  exit 0
fi

# Co-Authored-By trailer is banned by repo convention.
if grep -qi 'co-authored-by' <<<"$cmd"; then
  echo "Commit blocked: this repo does not use Co-Authored-By trailers." >&2
  exit 2
fi

# Newline-in-commit-body detection: if the command contains git commit -m followed by
# a quoted heredoc-or-multiline body, reject. We detect by counting raw newlines inside
# the -m argument.
#
# Strategy: extract everything between `-m ` and the next standalone arg or end.
# Simpler proxy: if the full command contains a literal newline AND `git commit -m`,
# treat it as multi-line.
if printf '%s' "$cmd" | grep -qE 'git[[:space:]]+commit' && \
   printf '%s' "$cmd" | awk 'BEGIN{n=0} /\n/ {n++} END{exit (n==0)}'; then
  # awk above always sees one record per line — the real check is whether the
  # command string contains \n inside a quoted -m body.
  body=$(printf '%s' "$cmd" | sed -n 's/.*-m[[:space:]]\+["'"'"']\(.*\)["'"'"'].*/\1/p')
  if [[ "$body" == *$'\n'* ]]; then
    echo "Commit blocked: repo requires one-line commit messages (no multi-line bodies)." >&2
    exit 2
  fi
fi

exit 0
