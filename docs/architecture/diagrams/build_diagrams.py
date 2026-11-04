"""Build the plain-English "How Spot works" diagrams.

One layout spec per diagram -> three outputs:
  <name>.png          hand-drawn style picture (PIL + Kalam font), 2x scale
  <name>.excalidraw   editable file: open at excalidraw.com or the VS Code Excalidraw extension
  <name>.mcp.json     element list for the Excalidraw MCP view (written to --mcp-dir if given)

Run: python3 docs/architecture/diagrams/build_diagrams.py [--mcp-dir DIR]
Edit the SPECS below, re-run, commit the PNG + .excalidraw together.
"""
import argparse
import json
import math
import os
import random

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SCALE = 2

# kind -> (fill, stroke, dark text colour for headings)
KINDS = {
    "hear":       ("#a5d8ff", "#4a9eed", "#1c4f8a"),
    "understand": ("#d0bfff", "#8b5cf6", "#4c2a9e"),
    "decide":     ("#fff3bf", "#f59e0b", "#8a5200"),
    "act":        ("#b2f2bb", "#22c55e", "#15803d"),
    "safety":     ("#ffc9c9", "#ef4444", "#a61b1b"),
    "note":       ("#f1f3f5", "#868e96", "#343a40"),
}
ZONES = {"hear": "#e7f5ff", "understand": "#f3f0ff", "decide": "#fff9db", "act": "#ebfbee",
         "safety": "#fff5f5"}
INK = "#1e1e1e"
MUTED = "#5c636a"


# ---------------------------------------------------------------- spec helpers
def box(id, x, y, w, h, kind, title, body="", size=22):
    return dict(t="box", id=id, x=x, y=y, w=w, h=h, kind=kind, title=title, body=body, size=size)


def zone(id, x, y, w, h, kind, label=""):
    return dict(t="zone", id=id, x=x, y=y, w=w, h=h, kind=kind, label=label)


def text(id, x, y, s, size=22, color=INK, bold=False, w=None):
    return dict(t="text", id=id, x=x, y=y, s=s, size=size, color=color, bold=bold, w=w)


def arrow(id, pts, label="", color=INK, dashed=False, size=18):
    return dict(t="arrow", id=id, pts=pts, label=label, color=color, dashed=dashed, size=size)


# ---------------------------------------------------------------- diagrams
def level0():
    W = 1150
    e = [
        text("t", 20, 10, "How Spot works", 40, bold=True),
        text("st", 20, 68, "Spot listens and looks, an AI decides what to do, then Spot talks and moves.", 22, MUTED),
        arrow("loop", [(1040, 190), (1040, 145), (150, 145), (150, 190)], "then it listens again", MUTED, True),
        box("b1", 20, 190, 250, 210, "hear", "1. Hears and sees", "a ring of microphones and five cameras around its body"),
        box("b2", 320, 190, 250, 210, "understand", "2. Understands", "turns your words into text and camera pictures into a description"),
        box("b3", 620, 190, 250, 210, "decide", "3. Decides", "an AI brain picks a reply and an action from a fixed list"),
        box("b4", 920, 190, 250, 210, "act", "4. Acts", "talks, walks, dances, and turns its gripper toward you"),
        arrow("a1", [(270, 295), (320, 295)]),
        arrow("a2", [(570, 295), (620, 295)]),
        arrow("a3", [(870, 295), (920, 295)]),
        arrow("s1", [(90, 400), (90, 480)], "", "#ef4444"),
        box("sf", 20, 480, W, 130, "safety", "Safety listener: always on, even when Spot is muted",
            "It hears \"stop\" or \"freeze\" and Spot stands still and tucks its arm right away, skipping the AI."),
        arrow("s2", [(1100, 480), (1100, 400)], "stops it", "#ef4444"),
        text("f1", 20, 640, "Spot only does what it is programmed to do. The AI can only choose from a fixed list of actions,", 20, MUTED),
        text("f2", 20, 670, "and Spot checks for space before it dances. A person with a laptop and an emergency stop button stays nearby.", 20, MUTED),
    ]
    return e, (1200, 900), (-15, -10)


def level1():
    e = [
        text("t", 20, 0, "How Spot works: the parts", 40, bold=True),
        text("st", 20, 56, "Left to right: what Spot senses, how it makes sense of it, what it decides, what it does.", 22, MUTED),
        zone("zh", 10, 100, 350, 770, "hear", "HEAR AND SEE"),
        zone("zu", 390, 100, 370, 770, "understand", "UNDERSTAND"),
        zone("zd", 790, 100, 370, 770, "decide", "DECIDE"),
        zone("za", 1190, 100, 370, 770, "act", "ACT"),
        # hear
        box("h1", 40, 150, 290, 200, "hear", "Microphone ring", "hears voices and which direction they come from"),
        box("h2", 40, 545, 290, 140, "hear", "Five body cameras", "see all around Spot"),
        box("h3", 40, 715, 290, 140, "hear", "Spot's own sensors", "nearby obstacles, printed markers, battery"),
        # understand
        box("u1", 420, 150, 310, 110, "understand", "Name listener", "waits for \"Hey Spot\" or \"Excuse me Spot\""),
        box("u2", 420, 290, 310, 110, "understand", "Who is talking?", "locks onto your direction (and voice, optional)"),
        box("u3", 420, 430, 310, 100, "understand", "Speech to text", "your words become text (Parakeet)"),
        box("u4", 420, 545, 310, 140, "understand", "Vision", "describes what it sees (Gemma), finds objects and people (YOLO)"),
        # decide
        box("d1", 820, 290, 310, 240, "decide", "AI brain (Gemma 4)",
            "picks a reply and actions from a fixed list. Knows about the event and Dartmouth. Runs on a Jetson computer, not the cloud."),
        box("d2", 820, 640, 310, 170, "decide", "Action checker", "checks for space before dancing, limits walking speed"),
        # act
        box("a1", 1220, 150, 310, 120, "act", "Speaks", "natural voice, with a backup voice that works offline"),
        box("a2", 1220, 300, 310, 120, "act", "Arm and gripper", "the gripper moves like a mouth and turns toward you"),
        box("a3", 1220, 450, 310, 140, "act", "Moves", "walk, sit, stand, dance, follow you, go to a place"),
        box("a4", 1220, 620, 310, 120, "act", "Light ring and beeps", "show when Spot is listening, thinking, speaking"),
        # flow
        arrow("e1", [(330, 205), (420, 205)]),
        arrow("e2", [(575, 260), (575, 290)]),
        arrow("e3", [(575, 400), (575, 430)]),
        arrow("e4", [(730, 480), (820, 440)]),
        arrow("e5", [(330, 615), (420, 615)]),
        arrow("e6", [(730, 615), (775, 615), (775, 490), (820, 490)]),
        arrow("e7", [(330, 785), (820, 755)]),
        arrow("e8", [(975, 530), (975, 640)]),
        arrow("e9", [(1130, 330), (1175, 330), (1175, 210), (1220, 210)], "words"),
        arrow("e10", [(1130, 720), (1175, 720), (1175, 520), (1220, 520)], "actions"),
        arrow("e11", [(1375, 270), (1375, 300)], "", MUTED, True),
        # safety
        arrow("s1", [(40, 250), (20, 250), (20, 950), (40, 950)], "", "#ef4444"),
        box("sf", 40, 900, 1490, 130, "safety", "Safety listener: always on, even when Spot is muted",
            "hears \"stop\" or \"freeze\" and makes Spot stand still and tuck its arm, skipping everything above"),
        arrow("s2", [(1530, 950), (1560, 950), (1560, 520), (1530, 520)], "", "#ef4444"),
        text("f1", 40, 1050, "Backups: stop words in the text also stop Spot before the AI brain; the operator laptop and an emergency stop button can always stop it.", 20, MUTED),
    ]
    return e, (1600, 1200), (-20, -20)


def walk_describe():
    e = [
        text("t", 20, 0, "Example: \"Hey Spot, what do you see?\"", 38, bold=True),
        text("st", 20, 56, "What happens, step by step, in a few seconds.", 22, MUTED),
        box("s1", 20, 130, 260, 180, "hear", "1. You ask", "\"Hey Spot, what do you see?\""),
        box("s2", 320, 130, 260, 180, "understand", "2. Spot hears its name", "the light ring points at you and Spot listens to you"),
        box("s3", 620, 130, 260, 180, "understand", "3. Speech to text", "your words become text"),
        box("s4", 920, 130, 260, 180, "decide", "4. AI brain decides", "\"describe the view\" (light ring breathes amber)"),
        box("s5", 920, 420, 260, 180, "hear", "5. Camera", "the front camera takes a picture"),
        box("s6", 620, 420, 260, 180, "understand", "6. Vision", "the vision model describes the picture"),
        box("s7", 320, 420, 260, 180, "act", "7. Spot answers", "says it out loud, the gripper moves like a mouth (ring turns blue)"),
        box("s8", 20, 420, 260, 180, "act", "8. Ready again", "keep talking: no need to say \"Hey Spot\" for about 20 seconds"),
        arrow("a1", [(280, 220), (320, 220)]),
        arrow("a2", [(580, 220), (620, 220)]),
        arrow("a3", [(880, 220), (920, 220)]),
        arrow("a4", [(1050, 310), (1050, 420)]),
        arrow("a5", [(920, 510), (880, 510)]),
        arrow("a6", [(620, 510), (580, 510)]),
        arrow("a7", [(320, 510), (280, 510)]),
    ]
    return e, (1200, 900), (-10, -20)


def walk_stop():
    e = [
        text("t", 20, 0, "Example: \"Stop!\"", 38, bold=True),
        text("st", 20, 56, "Stopping never waits for the AI. Three separate ways, fastest first.", 22, MUTED),
        box("p1", 20, 140, 260, 160, "hear", "You say", "\"stop\" or \"freeze\""),
        box("p2", 360, 140, 420, 160, "safety", "1. Safety listener",
            "hears every sound before anything else, even while Spot is talking or muted"),
        box("out", 860, 140, 300, 470, "act", "Spot stands still",
            "stops walking or dancing, tucks its arm, closes its gripper"),
        arrow("a1", [(280, 220), (360, 220)]),
        arrow("a2", [(780, 220), (860, 220)], "", "#ef4444"),
        box("q1", 360, 340, 420, 130, "understand", "2. Backup: stop-word check",
            "if the word reaches speech to text, Spot stops before the AI brain"),
        arrow("b1", [(150, 300), (150, 405), (360, 405)], "", MUTED, True),
        arrow("b2", [(780, 405), (860, 405)], "", MUTED, True),
        box("q2", 360, 500, 420, 110, "note", "3. The operator",
            "a laptop command, or the emergency stop button that cuts motor power"),
        arrow("b3", [(780, 555), (860, 555)], "", MUTED, True),
        text("f1", 20, 650, "Spot never says these words itself, so its own voice cannot stop it by accident.", 20, MUTED),
    ]
    return e, (1200, 900), (-10, -20)


SPECS = {
    "how-spot-works-level0": level0,
    "how-spot-works-level1": level1,
    "example-what-do-you-see": walk_describe,
    "example-stop": walk_stop,
}


# ---------------------------------------------------------------- text layout
def font(size, bold=False):
    name = "Kalam-Bold.ttf" if bold else "Kalam-Regular.ttf"
    try:
        return ImageFont.truetype(os.path.join(HERE, "fonts", name), int(size * SCALE))
    except OSError:
        return ImageFont.load_default()


def text_w(s, size, bold=False):
    f = font(size, bold)
    l, _, r, _ = f.getbbox(s)
    return (r - l) / SCALE


def wrap(s, size, max_w, bold=False):
    lines, cur = [], ""
    for word in s.split():
        trial = (cur + " " + word).strip()
        if cur and text_w(trial, size, bold) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def box_lines(el):
    """[(line, size, bold)] for a box, title bold then body regular."""
    pad = 22
    size = el["size"]
    out = [(ln, size, True) for ln in wrap(el["title"], size, el["w"] - 2 * pad, True)]
    if el["body"]:
        bsize = size - 3
        out += [(ln, bsize, False) for ln in wrap(el["body"], bsize, el["w"] - 2 * pad)]
    return out


LINE_H = 1.3


# ---------------------------------------------------------------- PNG renderer
def wobble(rng, p, amt=1.2):
    return (p[0] + rng.uniform(-amt, amt), p[1] + rng.uniform(-amt, amt))


def rough_line(d, rng, a, b, color, width):
    """Two slightly offset strokes, like a marker pen (Excalidraw roughness 1)."""
    for _ in range(2):
        pts = []
        n = max(2, int(math.dist(a, b) / 60) + 1)
        for i in range(n + 1):
            t = i / n
            pts.append(wobble(rng, (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t), 0.9))
        d.line([(x * SCALE, y * SCALE) for x, y in pts], fill=color, width=width, joint="curve")


def rough_rrect(d, rng, x, y, w, h, color, width, r=16):
    # straight sides with small arcs at the corners
    sides = [((x + r, y), (x + w - r, y)), ((x + w, y + r), (x + w, y + h - r)),
             ((x + w - r, y + h), (x + r, y + h)), ((x, y + h - r), (x, y + r))]
    for a, b in sides:
        rough_line(d, rng, a, b, color, width)
    for cx, cy, start in [(x + r, y + r, 180), (x + w - r, y + r, 270),
                          (x + w - r, y + h - r, 0), (x + r, y + h - r, 90)]:
        for j in range(2):
            o = rng.uniform(-0.8, 0.8)
            d.arc([(cx - r + o) * SCALE, (cy - r + o) * SCALE, (cx + r + o) * SCALE, (cy + r + o) * SCALE],
                  start, start + 90, fill=color, width=width)


def draw_text_lines(d, lines, cx, top, color_title, color_body):
    y = top
    for ln, size, bold in lines:
        f = font(size, bold)
        l, t, r, b = f.getbbox(ln)
        d.text((cx * SCALE - (r - l) / 2 - l, y * SCALE), ln, font=f,
               fill=color_title if bold else color_body)
        y += size * LINE_H


def content_bbox(elements, pad=40):
    """Canvas size and origin that fit every element plus padding."""
    xs, ys = [], []
    for el in elements:
        if el["t"] in ("box", "zone"):
            xs += [el["x"], el["x"] + el["w"]]
            ys += [el["y"], el["y"] + el["h"]]
        elif el["t"] == "arrow":
            xs += [p[0] for p in el["pts"]]
            ys += [p[1] for p in el["pts"]]
        elif el["t"] == "text":
            xs += [el["x"], el["x"] + text_w(el["s"], el["size"], el["bold"])]
            ys += [el["y"], el["y"] + el["size"] * 1.6]
    x0, y0 = min(xs) - pad, min(ys) - pad
    return (int(max(xs) + pad - x0), int(max(ys) + pad - y0)), (x0, y0)


def render_png(name, elements):
    (W, H), (ox, oy) = content_bbox(elements)
    rng = random.Random(name)
    img = Image.new("RGB", (W * SCALE, H * SCALE), "white")
    d = ImageDraw.Draw(img)

    def sh(el):  # shift into canvas coordinates
        return el["x"] - ox, el["y"] - oy

    # zones first, then boxes, then arrows, then free text
    for el in elements:
        if el["t"] == "zone":
            x, y = sh(el)
            d.rounded_rectangle([x * SCALE, y * SCALE, (x + el["w"]) * SCALE, (y + el["h"]) * SCALE],
                                radius=20 * SCALE, fill=ZONES[el["kind"]])
            if el["label"]:
                f = font(22, True)
                d.text(((x + 16) * SCALE, (y + 8) * SCALE), el["label"], font=f, fill=KINDS[el["kind"]][2])
    for el in elements:
        if el["t"] == "box":
            x, y = sh(el)
            fill, stroke, dark = KINDS[el["kind"]]
            d.rounded_rectangle([x * SCALE, y * SCALE, (x + el["w"]) * SCALE, (y + el["h"]) * SCALE],
                                radius=16 * SCALE, fill=fill)
            rough_rrect(d, rng, x, y, el["w"], el["h"], stroke, 2 * SCALE)
            lines = box_lines(el)
            total = sum(s * LINE_H for _, s, _ in lines)
            draw_text_lines(d, lines, x + el["w"] / 2, y + (el["h"] - total) / 2 - 4, dark, INK)
    for el in elements:
        if el["t"] == "arrow":
            pts = [(px - ox, py - oy) for px, py in el["pts"]]
            for a, b in zip(pts, pts[1:]):
                if el["dashed"]:
                    seg = math.dist(a, b)
                    n = max(1, int(seg / 14))
                    for i in range(0, n, 2):
                        t0, t1 = i / n, min(1, (i + 1) / n)
                        p0 = (a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0)
                        p1 = (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)
                        d.line([(p0[0] * SCALE, p0[1] * SCALE), (p1[0] * SCALE, p1[1] * SCALE)],
                               fill=el["color"], width=2 * SCALE)
                else:
                    rough_line(d, rng, a, b, el["color"], 2 * SCALE)
            # arrowhead on the last segment
            (x1, y1), (x2, y2) = pts[-2], pts[-1]
            ang = math.atan2(y2 - y1, x2 - x1)
            for s in (-1, 1):
                hx = x2 - 16 * math.cos(ang + s * 0.45)
                hy = y2 - 16 * math.sin(ang + s * 0.45)
                d.line([(x2 * SCALE, y2 * SCALE), (hx * SCALE, hy * SCALE)], fill=el["color"], width=2 * SCALE)
            if el["label"]:
                # label at the middle of the longest segment
                segs = list(zip(pts, pts[1:]))
                a, b = max(segs, key=lambda s: math.dist(*s))
                mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
                f = font(el["size"])
                l, t, r, bb = f.getbbox(el["label"])
                tw, th = (r - l) / SCALE, (bb - t) / SCALE
                d.rectangle([(mx - tw / 2 - 6) * SCALE, (my - th / 2 - 6) * SCALE,
                             (mx + tw / 2 + 6) * SCALE, (my + th / 2 + 8) * SCALE], fill="white")
                d.text(((mx - tw / 2) * SCALE - l, (my - th / 2) * SCALE - t), el["label"], font=f, fill=INK)
    for el in elements:
        if el["t"] == "text":
            x, y = sh(el)
            d.text((x * SCALE, y * SCALE), el["s"], font=font(el["size"], el["bold"]), fill=el["color"])
    img.save(os.path.join(HERE, name + ".png"), optimize=True)


# ---------------------------------------------------------------- Excalidraw file
def _base(id, type_, x, y, w, h, seed, **kw):
    el = dict(id=id, type=type_, x=x, y=y, width=w, height=h, angle=0, strokeColor=INK,
              backgroundColor="transparent", fillStyle="solid", strokeWidth=2, strokeStyle="solid",
              roughness=1, opacity=100, groupIds=[], frameId=None, roundness=None, seed=seed,
              version=1, versionNonce=seed + 1, isDeleted=False, boundElements=None,
              updated=1, link=None, locked=False)
    el.update(kw)
    return el


def _text_el(id, x, y, s, size, color, seed, align="left", groups=()):
    lines = s.split("\n")
    w = max(text_w(ln, size) for ln in lines) * 1.15 + 4  # Virgil runs a bit wider than Kalam
    h = size * 1.25 * len(lines)
    return _base(id, "text", x, y, w, h, seed, strokeColor=color, text=s, originalText=s,
                 fontSize=size, fontFamily=1, textAlign=align, verticalAlign="top",
                 containerId=None, lineHeight=1.25, baseline=int(size * 0.9), autoResize=True,
                 groupIds=list(groups))


def to_excalidraw(elements):
    out, seed = [], 1000
    for el in elements:
        seed += 7
        if el["t"] == "zone":
            out.append(_base(el["id"], "rectangle", el["x"], el["y"], el["w"], el["h"], seed,
                             strokeColor="transparent", backgroundColor=ZONES[el["kind"]],
                             roundness={"type": 3}))
            if el["label"]:
                out.append(_text_el(el["id"] + "_l", el["x"] + 16, el["y"] + 8, el["label"], 22,
                                    KINDS[el["kind"]][2], seed + 3))
        elif el["t"] == "box":
            fill, stroke, dark = KINDS[el["kind"]]
            g = ["g_" + el["id"]]
            out.append(_base(el["id"], "rectangle", el["x"], el["y"], el["w"], el["h"], seed,
                             strokeColor=stroke, backgroundColor=fill, roundness={"type": 3}, groupIds=g))
            # Virgil is wider than Kalam: wrap narrower so text stays inside in Excalidraw
            title = wrap(el["title"], el["size"], (el["w"] - 44) / 1.15, True)
            body = wrap(el["body"], el["size"] - 3, (el["w"] - 44) / 1.15) if el["body"] else []
            th = el["size"] * 1.25 * len(title)
            bh = (el["size"] - 3) * 1.25 * len(body)
            top = el["y"] + (el["h"] - th - bh) / 2
            t = _text_el(el["id"] + "_t", 0, top, "\n".join(title), el["size"], dark, seed + 3, "center", g)
            t["x"] = el["x"] + (el["w"] - t["width"]) / 2
            out.append(t)
            if body:
                b = _text_el(el["id"] + "_b", 0, top + th, "\n".join(body), el["size"] - 3, INK, seed + 5,
                             "center", g)
                b["x"] = el["x"] + (el["w"] - b["width"]) / 2
                out.append(b)
        elif el["t"] == "arrow":
            x0, y0 = el["pts"][0]
            pts = [[px - x0, py - y0] for px, py in el["pts"]]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            out.append(_base(el["id"], "arrow", x0, y0, max(xs) - min(xs), max(ys) - min(ys), seed,
                             strokeColor=el["color"], strokeStyle="dashed" if el["dashed"] else "solid",
                             points=pts, startArrowhead=None, endArrowhead="arrow",
                             startBinding=None, endBinding=None, lastCommittedPoint=None,
                             roundness={"type": 2} if len(pts) == 2 else None))
            if el["label"]:
                segs = list(zip(el["pts"], el["pts"][1:]))
                a, b = max(segs, key=lambda s: math.dist(*s))
                mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
                t = _text_el(el["id"] + "_l", 0, my - 12, el["label"], el["size"], INK, seed + 3, "center")
                t["x"] = mx - t["width"] / 2
                out.append(t)
        elif el["t"] == "text":
            out.append(_text_el(el["id"], el["x"], el["y"], el["s"], el["size"], el["color"], seed))
    return {"type": "excalidraw", "version": 2, "source": "build_diagrams.py", "elements": out,
            "appState": {"viewBackgroundColor": "#ffffff", "gridSize": None}, "files": {}}


# ---------------------------------------------------------------- MCP view
def to_mcp(elements, canvas, origin):
    out = [{"type": "cameraUpdate", "width": canvas[0], "height": canvas[1], "x": origin[0], "y": origin[1]}]
    for el in elements:
        if el["t"] == "zone":
            out.append({"type": "rectangle", "id": el["id"], "x": el["x"], "y": el["y"], "width": el["w"],
                        "height": el["h"], "backgroundColor": ZONES[el["kind"]], "fillStyle": "solid",
                        "strokeColor": "transparent", "roundness": {"type": 3}})
            if el["label"]:
                out.append({"type": "text", "id": el["id"] + "_l", "x": el["x"] + 16, "y": el["y"] + 8,
                            "text": el["label"], "fontSize": 22, "strokeColor": KINDS[el["kind"]][2]})
        elif el["t"] == "box":
            fill, stroke, dark = KINDS[el["kind"]]
            label = el["title"] + ("\n" + el["body"] if el["body"] else "")
            out.append({"type": "rectangle", "id": el["id"], "x": el["x"], "y": el["y"], "width": el["w"],
                        "height": el["h"], "backgroundColor": fill, "fillStyle": "solid",
                        "strokeColor": stroke, "roundness": {"type": 3},
                        "label": {"text": label, "fontSize": el["size"] - 2}})
        elif el["t"] == "arrow":
            x0, y0 = el["pts"][0]
            pts = [[px - x0, py - y0] for px, py in el["pts"]]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            a = {"type": "arrow", "id": el["id"], "x": x0, "y": y0, "width": max(xs) - min(xs),
                 "height": max(ys) - min(ys), "points": pts, "strokeColor": el["color"],
                 "endArrowhead": "arrow"}
            if el["dashed"]:
                a["strokeStyle"] = "dashed"
            if el["label"]:
                a["label"] = {"text": el["label"], "fontSize": el["size"]}
            out.append(a)
        elif el["t"] == "text":
            out.append({"type": "text", "id": el["id"], "x": el["x"], "y": el["y"], "text": el["s"],
                        "fontSize": el["size"], "strokeColor": el["color"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcp-dir", help="also write <name>.mcp.json element lists here")
    args = ap.parse_args()
    for name, fn in SPECS.items():
        elements, canvas, origin = fn()
        ids = [e["id"] for e in elements]
        assert len(ids) == len(set(ids)), f"{name}: duplicate ids"
        render_png(name, elements)
        with open(os.path.join(HERE, name + ".excalidraw"), "w") as f:
            json.dump(to_excalidraw(elements), f, indent=1)
        if args.mcp_dir:
            with open(os.path.join(args.mcp_dir, name + ".mcp.json"), "w") as f:
                json.dump(to_mcp(elements, canvas, origin), f, separators=(",", ":"))
        print("built", name)


if __name__ == "__main__":
    main()
