---
name: block-grep-secret-value
enabled: true
event: bash
action: block
conditions:
  - field: command
    operator: regex_match
    pattern: \bgrep\b(?!\s+(-[a-zA-Z]*[lLcq]|--count|--files-with-matches|--files-without-match|--quiet))
  - field: command
    operator: regex_match
    pattern: \.env(?!\.example\b)(\.[a-z0-9_-]+)?\b|\bcredentials\.json\b|\bservice[-_]account[^/\s]*\.json\b|\.(pem|key|p12|pfx|jks|keystore|kdbx)\b|\b(id_rsa|id_ed25519|id_ecdsa|id_dsa)\b|\.aws/credentials\b|\.aws/config\b|\.ssh/(?!.*\.pub\b)|\.docker/config\.json\b|\.config/gh/hosts\.yml\b|\.netrc\b|\.git-credentials\b|\.npmrc\b|\.pypirc\b|\bsecrets?\.ya?ml\b|\b(bash|zsh|python)_history\b|\.viminfo\b
---

🚨 **BLOCKED: `grep` without `-l`/`-L`/`-c`/`-q` would echo the matched line (including secret values) to the transcript.**

Transcript is append-only. Leaked values must be rotated.

**Safe grep flags:**
```bash
grep -l PATTERN <file>    # filename if match (no value)
grep -L PATTERN <file>    # filename if no match
grep -c PATTERN <file>    # match count only
grep -q PATTERN <file>    # exit code only
```

**Length-only check:**
```bash
awk -F= '/^KEY=/ {print length($2)}' <file>
```
