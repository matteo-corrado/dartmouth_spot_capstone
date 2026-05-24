---
name: block-secret-file-read
enabled: true
event: bash
action: block
conditions:
  - field: command
    operator: regex_match
    pattern: \b(cat|head|tail|less|more|bat|view|xxd|hexdump|strings|nl|tac|rev)\b
  - field: command
    operator: regex_match
    pattern: \.env(?!\.example\b)(\.[a-z0-9_-]+)?\b|\bcredentials\.json\b|\bservice[-_]account[^/\s]*\.json\b|\.(pem|key|p12|pfx|jks|keystore|kdbx)\b|\b(id_rsa|id_ed25519|id_ecdsa|id_dsa)\b|\.aws/credentials\b|\.aws/config\b|\.ssh/(?!.*\.pub\b)|\.docker/config\.json\b|\.config/gh/hosts\.yml\b|\.netrc\b|\.git-credentials\b|\.npmrc\b|\.pypirc\b|\bsecrets?\.ya?ml\b|\b(bash|zsh|python)_history\b|\.viminfo\b|/proc/[^/\s]+/environ\b
---

🚨 **BLOCKED: command would echo secret file contents to the transcript.**

The transcript is append-only — leaked values cannot be scrubbed and the key must be rotated.

**Use presence/metadata/length-only alternatives:**
```bash
grep -l PATTERN <file> && echo present    # filename only
grep -c PATTERN <file>                    # match count
test -f <file> && echo present
ls -la <file>                             # metadata only
wc -l <file>                              # line count only
stat <file>                               # metadata only
awk -F= '/^KEY=/ {print length($2)}' <file>  # value length only
source <file>                             # inject into env (no echo)
```
