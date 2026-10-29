"""BD stitch_front_images, edited: capture once (or load saved frames), render
in a hidden window on DISPLAY :0, read pixels back, save PNG.
BD classes and shaders are unchanged (imported from the copied example)."""
import argparse, os, pathlib, time
import pygame
from OpenGL.GL import *
from PIL import Image
from dotenv import load_dotenv
import bosdyn.client
import bosdyn.client.util
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient, build_image_request
import stitch_front_images as bd

HERE = pathlib.Path(__file__).parent
load_dotenv(pathlib.Path("/home/spotdog/spot/dartmouth_spot_capstone") / ".env")

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="192.168.50.3")
ap.add_argument("--frames", default=str(HERE / "frames.bin"))
ap.add_argument("--capture", action="store_true", help="grab fresh frames from robot and save them")
a = ap.parse_args()

if a.capture:
    sdk = bosdyn.client.create_standard_sdk("front_cam_stitch_once")
    robot = sdk.create_robot(a.host)
    bosdyn.client.util.authenticate(robot)  # reads login from environment
    robot.time_sync.wait_for_sync()
    ic = robot.ensure_client(ImageClient.default_service_name)
    resp = image_pb2.GetImageResponse()
    resp.image_responses.extend(ic.get_image([build_image_request(s, quality_percent=90)
                                              for s in ["frontright_fisheye_image", "frontleft_fisheye_image"]]))
    pathlib.Path(a.frames).write_bytes(resp.SerializeToString())
    print("saved frames", a.frames)

images = bd.load_get_image_response_from_binary_file(a.frames).image_responses
fr = fl = None
for im in images:
    if im.source.name == "frontright_fisheye_image":
        fr = bd.ImagePreppedForOpenGL(im)
    elif im.source.name == "frontleft_fisheye_image":
        fl = bd.ImagePreppedForOpenGL(im)

os.chdir(HERE)  # shaders live next to this file
pygame.init()
display = (1080, 720)
pygame.display.set_mode(display, pygame.DOUBLEBUF | pygame.OPENGL | pygame.HIDDEN)
program = bd.CompiledShader(open("shader_vert.glsl").read(), open("shader_frag.glsl").read())
program.update_images(fr, fl)  # same argument order as BD's loop
cam = bd.StitchingCamera(fr, fl)
t = time.time()
glClearColor(0, 0, 0, 0)
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
bd.draw_routine(display, program, cam)
glFinish()
px = glReadPixels(0, 0, display[0], display[1], GL_RGB, GL_UNSIGNED_BYTE)
print(f"render+readback {1000 * (time.time() - t):.1f} ms")
img = Image.frombytes("RGB", display, px).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
img.save(HERE / "bd_stitch.png")
print("saved", HERE / "bd_stitch.png")
