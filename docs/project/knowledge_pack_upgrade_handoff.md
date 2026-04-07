# Handoff: upgrade the Spot tour-guide knowledge packs

You are being asked to improve two markdown files that get loaded into a small
local LLM's system prompt. The LLM runs on a Boston Dynamics Spot robot acting
as a tour guide for **Thayer School of Engineering at Dartmouth College**.
This brief gives you everything you need to do the work without having to
ask follow-up questions.

---

## What this is for

- **Robot**: Boston Dynamics Spot, voice-controlled, on a Jetson AGX Orin.
- **LLM stack**: Ollama on the Jetson. Currently serving `qwen2.5:7b` (text)
  and `qwen2.5vl:7b` (vision). The text model is what reads these knowledge
  packs. A move to `gemma4:e4b` is planned per
  `docs/project/upgrade-plan.md` Phase 2 — write content that will work for
  any small open-weight model with a 32K+ context window.
- **Audience**: visitors to Thayer School, including children, parents,
  prospective students, and alumni. Output is **spoken aloud** via TTS, so
  it must sound natural read out loud.
- **Tone**: warm, conversational, kid-friendly. Imagine a friendly docent
  talking to an 11-year-old's family. Avoid jargon, marketing-speak, and
  long paragraphs.
- **Hard rules**: facts only (no invention), child-appropriate (no politics
  or controversies), say "I don't know" rather than guessing.

## The two files you're upgrading

### File 1 — `src/voice_control/knowledge/thayer_knowledge.md`
General background pack. Always loaded into the system prompt. Used to answer
visitor questions like "When was Thayer founded?", "What programs are
offered?", "Who's a famous alum?", "Tell me about Dartmouth."

**Current state**: 271 lines / 3,349 words / ~4,500 tokens. Sourced almost
entirely from Wikipedia because the original drafting agent could not use
WebFetch — only WebSearch. Every fact has an inline `(source: ...)`
parenthetical.

### File 2 — `src/voice_control/knowledge/tour_route.md`
Tour route reference. Always loaded. Walks Spot through 16 ordered stops
through the Thayer buildings (MacLean → Cummings → ECSC), with talking
points for each stop. Cross-references saved navigation locations from
`locations.json`.

**Current state**: 258 lines / 1,725 words / ~2,300 tokens. Built from a
Dartmouth student tour guide's hand-written notes that were given to the
drafting assistant inline. Has a section at the bottom listing five
"open questions" that need authoritative answers.

## Combined size budget

| Limit | Tokens | Why |
|---|---|---|
| Current combined | ~6,800 | Where you start |
| **Soft target after upgrade** | 8,000–12,000 | Plenty of headroom for added facts |
| **Hard ceiling** | 15,000 | Beyond this, the system prompt eats too much context budget; the LLM needs room for conversation history and tool outputs |

KV-cache prefill on the Jetson means a fat system prompt is essentially
free **after** the first call (Ollama caches it), so larger is fine **as
long as it's high-signal content**, not filler. Don't pad. Don't repeat
yourself. Each fact should earn its tokens.

## Concerns with the current state

1. **Cummings Hall date contradiction (UNRESOLVED).** The student tour notes
   say Cummings was "built in 1939". Wikipedia (the agent's source) says
   "opened in 1946" and that it was "the first building constructed
   specifically for Thayer" (prior to that, Thayer occupied borrowed rooms
   in Wentworth, Reed, and Thornton Halls). Both can't be right.
   **Please find authoritative confirmation** on `engineering.dartmouth.edu`
   and reconcile. The friendly Spot script in `tour_route.md` currently
   says 1939 (matching the student notes); the background pack says 1946.
   They need to agree.

2. **ECSC completion date is missing from `thayer_knowledge.md`.** The
   student tour notes say ECSC "construction was completed in March of
   2022". The drafting agent could not confirm this from Wikipedia, so it
   was left out. Please verify on engineering.dartmouth.edu and add to the
   buildings section if confirmed.

3. **ECSC vs CESC vs ECCS naming.** The official building name is
   "Class of 1982 Engineering and Computer Science Center" — abbreviated
   ECSC. The saved location key in `locations.json` is `cesc_building`
   (clearly a typo or inconsistent legacy name). Don't try to fix
   `locations.json` — that's a code change. Just make sure the docs
   consistently use **ECSC** as the spoken name.

4. **Inline `(source: ...)` citations are visible to the LLM.** They're
   currently ~400 tokens of noise the model has to read every prompt.
   Two acceptable approaches:
   - Strip them entirely (smaller file, less verifiability)
   - Move them all to a single "Sources" footer section at the bottom of
     each file (verifiability preserved, less distracting to the LLM)
   Pick one and apply consistently across both files.

5. **The drafting agent could not use WebFetch.** All facts in
   `thayer_knowledge.md` came from WebSearch result snippets and Wikipedia.
   You should be able to use WebFetch directly on `engineering.dartmouth.edu`
   pages and pull richer, more authoritative content. Treat the current
   pack as a starting skeleton, not a finished product.

6. **Open questions in `tour_route.md` (bottom section) are unanswered.**
   Five things the original notes were ambiguous about. Please research
   each on Thayer's website / Dartmouth news / authoritative sources:
   - **T&T sessions** in ENGS 21 — what does "T&T" stand for?
   - **ERAS** (a Thayer research program) — what does it stand for, and a
     one-line description
   - **Spanos** — full name (likely "Spanos Auditorium" or similar) and
     anything notable beyond class size
   - **Allyn Lab** — anything beyond housing Dartmouth Formula Racing
   - **Magnuson Center** — exact official name (Magnuson Center for
     Entrepreneurship vs. Magnuson Family Center for Entrepreneurship?)

7. **The inline citations on Thayer-specific facts cite URLs the original
   agent never actually fetched** — only WebSearched. Re-verify each
   Thayer-domain citation by actually fetching the URL. If a fact can't
   be confirmed at the cited URL, find a new source or remove the fact.

## Specific improvements wanted

In rough priority order:

1. **Resolve the Cummings date** and the ECSC completion date.
2. **Answer the five open questions** in `tour_route.md` and remove that
   section.
3. **Re-verify Thayer-domain citations** by actually fetching them.
4. **Add current research areas** (not faculty names — those rot — but
   evergreen research themes per the engineering.dartmouth.edu/research
   pages). E.g., "biomedical imaging at the Optics in Medicine lab",
   "cold-region engineering and ice mechanics", "robotics and autonomy".
5. **Add named research labs and centers** that are referenced on
   Thayer's official site but missing from the current pack. The tour
   notes mention "level-by-level poster" of ECSC labs — try to populate
   a list of those labs by name with one-line descriptions each.
6. **Add a short "Practical visitor info" section** to
   `thayer_knowledge.md`: how to get to Thayer (Hanover NH, the
   approximate parking situation), when classes are in session
   (Dartmouth's quarter system / "D-Plan"), whether visitors can sit in
   on classes, etc. Keep it evergreen — no specific dates.
7. **Add famous alumni** if any can be authoritatively confirmed via
   engineering.dartmouth.edu/alumni or news.dartmouth.edu. The current
   pack has only Sylvanus Thayer (founder), Robert Fletcher (first
   dean), John P. Holdren, and William Kamkwamba. There are likely
   more. **Do not invent.** If you can't confirm, leave it.
8. **Trim or rewrite anything that sounds dry or academic.** The output
   gets read aloud to children. The current pack is okay but a few
   sections (research disciplines, research culture stats) read like
   bullet points off a brochure. They should sound more like a friendly
   guide explaining things.
9. **Remove the "How Spot should handle questions" section** at the
   bottom of `thayer_knowledge.md`. That guidance belongs in
   `llm_brain.py`'s `SYSTEM_PROMPT`, not in a knowledge file. (I'll
   move it to the system prompt during integration. Don't put it in
   either knowledge file.)
10. **Keep the "Notes on uncertainty and source contradictions"
    section** in `thayer_knowledge.md` but update it as you resolve
    things.

## What NOT to do

- **Do not invent facts.** Wrong facts said by a robot to a child's
  parents at a tour are embarrassing. If a fact isn't in an authoritative
  source, omit it.
- **Do not include current rankings, tuition, enrollment-this-year,
  course schedules, or any specific staff names that change** (deans
  excluded — they're stable enough for a few years). These rot fast.
- **Do not add political content, controversies, lawsuits, or anything
  not appropriate for kids.**
- **Do not exceed the 15,000 token combined ceiling.** Measure with
  `wc -w` (words × ~1.3 = tokens) before submitting.
- **Do not modify `locations.json`, `llm_brain.py`, or any code file.**
  You're only working on the two markdown files.
- **Do not change the basic structure of `tour_route.md`** — the 16
  ordered stops are calibrated to the actual Thayer building layout.
  You can enrich the talking points within each stop, but don't
  reorder, drop, or merge stops without flagging it explicitly.

## Output format

Return the two upgraded markdown files. For each file:

1. The full updated content
2. A short changelog at the top of your response (not in the file
   itself) listing what you changed, what you added, what you removed,
   and any open questions you couldn't resolve.
3. Empirical word count for each file so the user can verify the size
   constraint.

## Authoritative sources to use

Priority order — please use **WebFetch** (not just search) on these:

1. https://engineering.dartmouth.edu/about
2. https://engineering.dartmouth.edu/about/history
3. https://engineering.dartmouth.edu/about/facts
4. https://engineering.dartmouth.edu/academics
5. https://engineering.dartmouth.edu/undergraduate
6. https://engineering.dartmouth.edu/graduate
7. https://engineering.dartmouth.edu/research
8. https://engineering.dartmouth.edu/people (faculty)
9. https://engineering.dartmouth.edu/community
10. https://engineering.dartmouth.edu/visit
11. https://engineering.dartmouth.edu/design (project-based courses)
12. https://engineering.dartmouth.edu/design/cedc (Cook Engineering
    Design Center)
13. https://news.dartmouth.edu/ (search for "Thayer")
14. https://en.wikipedia.org/wiki/Thayer_School_of_Engineering
15. https://en.wikipedia.org/wiki/Dartmouth_College
16. https://en.wikipedia.org/wiki/Sylvanus_Thayer

Follow links into sub-pages where they reveal useful detail. Cross-check
between Thayer's official site and Wikipedia where they overlap. When
they disagree, **prefer the Thayer official site** and note the
disagreement in the changelog.
