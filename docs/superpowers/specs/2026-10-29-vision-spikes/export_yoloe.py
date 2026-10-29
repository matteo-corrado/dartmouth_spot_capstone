"""THROWAWAY: export yoloe-26l-seg with a baked indoor/lab/event noun list (run in export_env, cwd=models)."""
from ultralytics import YOLOE

NOUNS = """person, door, stairs, staircase, elevator, window, wall, floor, ceiling light, poster, sign, banner,
whiteboard, blackboard, projector screen, monitor, tv, laptop, computer, keyboard, mouse, phone, tablet,
table, desk, chair, office chair, stool, bench, couch, shelf, bookshelf, cabinet, drawer, locker, cart,
trash can, recycling bin, box, cardboard box, crate, backpack, bag, handbag, suitcase, briefcase,
bottle, water bottle, cup, mug, coffee cup, can, plate, bowl, food, pizza, snack, fridge, microwave, sink,
plant, potted plant, flower, tree, lamp, clock, fan, heater, fire extinguisher, exit sign, outlet,
cable, power strip, toolbox, tool, screwdriver, wrench, hammer, drill, scissors, tape, ladder, bucket,
robot, robot arm, drone, 3d printer, printer, oscilloscope, soldering iron, circuit board, battery,
helmet, safety glasses, gloves, jacket, coat, hat, cap, shoe, umbrella, glasses, book, notebook, paper,
pen, marker, folder, envelope, badge, lanyard, name tag, camera, tripod, microphone, speaker,
headphones, guitar, ball, toy, dog, cat, bicycle, scooter, skateboard, wheelchair, stroller,
car, truck, bus, traffic cone, barrier, rope, flag, balloon, cake, award, trophy, easel, podium, stage,
curtain, mirror, picture frame, painting, sculpture, vending machine, water fountain, coat rack, umbrella stand""".replace("\n", " ")
names = [n.strip() for n in NOUNS.split(",") if n.strip()]
print(len(names), "classes")
m = YOLOE("yoloe-26l-seg.pt")
m.set_classes(names)
p = m.export(format="onnx", imgsz=640, simplify=True, opset=17)
open("yoloe_names.txt", "w").write("\n".join(names))
print("OK", p)
