#!/usr/bin/env python3
"""Stage 2F P7 / V5 — verify Spot onboard AV LED control.

Connects, reports whether the robot has an AV system, lists existing
behaviors, then drives each voice state's pulse color for 2s. Run on the
Jetson with the robot powered (motors NOT required — AV needs no lease).

    spot-env/bin/python scripts/diag_av_leds.py --hostname <ROBOT_IP>
"""
import argparse
import time

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.audio_visual import AudioVisualClient
from bosdyn.util import now_sec


def main():
    # Delayed imports to avoid circular dep between client_mic and state_feedback.
    from src.voice_control.client_mic import VoiceState
    from src.voice_control.spot_leds import _build_pulse_behavior, _behavior_name
    from src.voice_control.state_feedback import led_color_for, LED_PERIOD_S

    p = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(p)
    args = p.parse_args()

    sdk = bosdyn.client.create_standard_sdk("av-led-diag")
    robot = sdk.create_robot(args.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    hw = robot.get_cached_hardware_hardware_configuration()
    print(f"has_audio_visual_system = {getattr(hw, 'has_audio_visual_system', None)}")

    av = robot.ensure_client(AudioVisualClient.default_service_name)
    try:
        print("existing behaviors:", av.list_behaviors())
    except Exception as e:
        print(f"list_behaviors failed: {e}")

    for st in VoiceState:
        r, g, b = led_color_for(st)
        name = _behavior_name(st)
        av.add_or_modify_behavior(name, _build_pulse_behavior(r, g, b, LED_PERIOD_S[st]))
        print(f"running {name} rgb=({r},{g},{b}) for 2s...")
        av.run_behavior(name, end_time_secs=now_sec() + 2.5)
        time.sleep(2.0)
        av.stop_behavior(name)

    print("done — if you saw blue/green/amber/cyan pulses, V5 PASSES.")


if __name__ == "__main__":
    main()
