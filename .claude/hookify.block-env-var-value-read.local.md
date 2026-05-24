---
name: block-env-var-value-read
enabled: true
event: bash
action: block
conditions:
  - field: command
    operator: regex_match
    pattern: (\bprintenv\b\s+\S*(?i:KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|PRIVATE|BEARER|API_KEY|AWS_(ACCESS|SECRET)|GCP_|AZURE_|_PAT|PAT_))|((?<![./\w])(env|printenv|set)\b\s*\|\s*grep\b(?!\s+-[a-zA-Z]*[lLcq]))|((?<![./\w])(env|printenv)\b\s*$)|(\becho\s+[\"']?\$\{?[A-Z_]*(?i:KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|PRIVATE|BEARER|API_KEY)[A-Z_]*(?![:+\-?=]))|(\bprintf\b[^|]*\$\{?[A-Z_]*(?i:KEY|SECRET|TOKEN|PASSWORD)[A-Z_]*(?![:+\-?=]))
---

🚨 **BLOCKED: command would echo a secret environment variable's VALUE to the transcript.**

Transcript is append-only — once a secret value lands, the key must be rotated.

**Presence-only alternatives:**
```bash
[ -n "${SECRET_KEY+x}" ] && echo set            # presence test (no value)
echo "${SECRET_KEY:+set}"                       # "set" if non-empty, else nothing
compgen -e | grep KEY                           # env var NAMES only
printenv | cut -d= -f1 | grep KEY               # NAMES only
env | wc -l                                     # count only
```

**Length check (no value):**
```bash
awk -v v="$SECRET_KEY" 'BEGIN {print length(v)}'
```
