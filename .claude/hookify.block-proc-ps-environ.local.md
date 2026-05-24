---
name: block-proc-ps-environ
enabled: true
event: bash
action: block
conditions:
  - field: command
    operator: regex_match
    pattern: (/proc/[^/\s]+/environ\b)|(\bps\s+(-?\w*[ae]\w*e\w*|-e\w*\s+\w*environ|aux?e\b|eww?\b))
---

🚨 **BLOCKED: command would read process environment (likely containing API keys / tokens) to stdout.**

`/proc/<pid>/environ` and `ps -e env`-style flags dump the process environment block, which on this Jetson includes API keys loaded via `source .env`.

**Safer alternatives:**
```bash
ps aux | head             # processes WITHOUT env
pgrep -af PATTERN         # PIDs + cmdline only, no env
ps -o pid,comm,args       # explicit columns (no environ)
```
