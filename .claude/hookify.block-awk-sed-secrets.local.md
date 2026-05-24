---
name: block-awk-sed-secrets
enabled: true
event: bash
action: block
conditions:
  - field: command
    operator: regex_match
    pattern: (\bawk\b[^|]*\bprint\b(?!\s+length))|(\bsed\b\s+-n\s+[^|]*\bp\b)|(\bcut\b\s+-[df])
  - field: command
    operator: regex_match
    pattern: \.env(?!\.example\b)(\.[a-z0-9_-]+)?\b|\bcredentials\.json\b|\bservice[-_]account[^/\s]*\.json\b|\.(pem|key|p12|pfx|jks|keystore|kdbx)\b|\b(id_rsa|id_ed25519|id_ecdsa|id_dsa)\b|\.aws/credentials\b|\.aws/config\b|\.ssh/(?!.*\.pub\b)|\.docker/config\.json\b|\.config/gh/hosts\.yml\b|\.netrc\b|\.git-credentials\b|\.npmrc\b|\.pypirc\b|\bsecrets?\.ya?ml\b|\b(bash|zsh|python)_history\b|\.viminfo\b
---

🚨 **BLOCKED: `awk print`, `sed -n /PAT/p`, or `cut -d/-f` on a secret file would echo values to the transcript.**

Allowed `awk` form (length-only):
```bash
awk -F= '/^KEY=/ {print length($2)}' <file>
```

Exit-code-only check:
```bash
awk -F= '/^KEY=/ {found=1} END {exit !found}' <file> && echo present
```
