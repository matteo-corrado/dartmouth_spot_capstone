"""THE nav thread (walk watch, GraphNav sequence, visual nav, follow, emote)
plus GraphNav localization and the come_back home waypoint.

Moved out of spot_dispatch globals ``_nav_thread``, ``_nav_stop_event``,
``_nav_mode``, ``_nav_lock`` and ``_home_waypoint``.
"""
import threading
import time

from bosdyn.client.frame_helpers import get_odom_tform_body

from src.spot.nav.graph_nav_utils import capped_travel_params


class NavRunner:
    def __init__(self, safety, connection):
        self._safety = safety
        self._conn = connection
        self._thread = None
        self._stop_event = None
        self._mode = None  # "follow", "graphnav", "visual_nav", "walk", "emote", or None
        self._lock = threading.Lock()  # guards the (_thread, _stop_event, _mode) triple
        self._home_waypoint = None

    def spawn(self, fn, mode=None, gen=None) -> bool:
        """Start fn(stop_event) as THE nav thread. The abort check and the
        registration happen under the nav lock, and stop bumps the safety gen
        before its cancel() takes the same lock, so a thread is either
        registered (and then cancelled by the stop) or never started: no
        orphan can keep moving."""
        with self._lock:
            if self._safety.aborted(gen):
                print("[Spot] Motion not started: a stop arrived")
                return False
            ev = threading.Event()
            t = threading.Thread(target=fn, args=(ev,), daemon=True)
            self._thread, self._stop_event, self._mode = t, ev, mode
            t.start()
        return True

    def cancel(self):
        """Cancel any in-progress navigation thread (sets its event, then joins
        for up to 2 s outside the registry lock)."""
        with self._lock:
            t, ev = self._thread, self._stop_event
            self._thread = None
            self._stop_event = None
            self._mode = None
            if ev:
                ev.set()
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.0)

    def is_navigating(self) -> bool:
        """Check if the robot is currently navigating (motor noise expected)."""
        return self._thread is not None and self._thread.is_alive()

    def is_follow_mode(self) -> bool:
        """Check if robot is in follow-person mode (lighter motor noise)."""
        return self._mode == "follow" and self.is_navigating()

    # ------------------------------------------------------------ GraphNav
    def ensure_localized(self, graph_nav_client, session) -> bool:
        """Check localization and attempt fiducial/waypoint init if needed."""
        localization = graph_nav_client.get_localization_state()
        if localization.localization.waypoint_id:
            print(f"[Spot] Localized at waypoint: {localization.localization.waypoint_id}")
            return True

        print("[Spot] Not localized. Attempting to localize...")
        return self.force_relocalize(graph_nav_client, session)

    def force_relocalize(self, graph_nav_client, session) -> bool:
        """Force re-localization using fiducials first, then waypoint fallback."""
        from bosdyn.api.graph_nav import nav_pb2, graph_nav_pb2
        import math

        # Try fiducial-based localization first (most accurate)
        try:
            print("[Spot] Trying fiducial-based localization...")
            empty_loc = nav_pb2.Localization()
            graph_nav_client.set_localization(
                initial_guess_localization=empty_loc,
                fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NEAREST
            )
            localization = graph_nav_client.get_localization_state()
            if localization.localization.waypoint_id:
                print(f"[Spot] Localized via fiducial at: {localization.localization.waypoint_id}")
                return True
            print("[Spot] Fiducial localization failed (no fiducials visible)")
        except Exception as e:
            print(f"[Spot] Fiducial localization error: {e}")

        # Fallback: waypoint-based localization
        if session is None:
            return False

        try:
            from bosdyn.client.robot_state import RobotStateClient

            robot_state_client = session["robot"].ensure_client(RobotStateClient.default_service_name)
            robot_state = robot_state_client.get_robot_state()
            current_odom_tform_body = get_odom_tform_body(
                robot_state.kinematic_state.transforms_snapshot).to_proto()

            graph = graph_nav_client.download_graph()
            if not graph.waypoints:
                print("[Spot] No waypoints in map. Please upload map first.")
                return False

            first_waypoint_id = graph.waypoints[0].id
            print(f"[Spot] Trying waypoint-based localization to: {first_waypoint_id}")

            loc_guess = nav_pb2.Localization()
            loc_guess.waypoint_id = first_waypoint_id
            loc_guess.waypoint_tform_body.rotation.w = 1.0

            graph_nav_client.set_localization(
                initial_guess_localization=loc_guess,
                max_distance=5.0,   # Allow larger search radius
                max_yaw=180.0 * math.pi / 180.0,  # Allow any orientation
                fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NO_FIDUCIAL,
                ko_tform_body=current_odom_tform_body
            )

            localization = graph_nav_client.get_localization_state()
            if localization.localization.waypoint_id:
                print(f"[Spot] Localized via waypoint at: {localization.localization.waypoint_id}")
                return True

            print("[Spot] Localization failed. Robot may not be near a known waypoint.")
            return False
        except Exception as e:
            print(f"[Spot] Localization error: {e}")
            return False

    def save_home_waypoint(self, graph_nav_client):
        """Save current localized waypoint as departure point for come_back."""
        try:
            loc = graph_nav_client.get_localization_state()
            if loc.localization.waypoint_id:
                self._home_waypoint = loc.localization.waypoint_id
        except Exception:
            pass

    def navigate_sequence(self, graph_nav_client, waypoint_ids, location_names,
                          stop_event, repeat=False, gen=None):
        """Navigate through a sequence of waypoints. Runs in a background thread.

        On STATUS_STUCK: retries up to MAX_STUCK_RETRIES with a new command ID.
        If still stuck after retries, skips to the next waypoint (multi-waypoint)
        or gives up (single waypoint).

        Returns True only when every waypoint was reached (natural finish), so
        the arm is restored only then. A stop since `gen` aborts before any
        navigate_to.
        """
        from bosdyn.api.graph_nav import graph_nav_pb2

        MAX_STUCK_RETRIES = 2  # Retry with fresh command ID before skipping

        pass_num = 0
        all_reached = True
        try:
            while True:
                pass_num += 1
                if repeat and pass_num > 1:
                    print(f"[Spot] Starting patrol pass #{pass_num}")

                for i, (wp_id, name) in enumerate(zip(waypoint_ids, location_names)):
                    if stop_event.is_set():
                        print("[Spot] Navigation cancelled")
                        return

                    print(f"[Spot] Navigating to '{name}' ({i+1}/{len(waypoint_ids)})...")
                    stuck_retries = 0
                    nav_to_cmd_id = None
                    waypoint_reached = False

                    while not stop_event.is_set():
                        if self._safety.aborted(gen):
                            print("[Spot] Navigation aborted by a stop")
                            return False
                        try:
                            nav_to_cmd_id = graph_nav_client.navigate_to(
                                wp_id, 10.0,
                                travel_params=capped_travel_params(),
                                command_id=nav_to_cmd_id
                            )
                        except Exception as nav_ex:
                            # navigate_to() raises RobotStuckError as exception
                            if "Stuck" in type(nav_ex).__name__ or "Stuck" in str(nav_ex):
                                stuck_retries += 1
                                if stuck_retries <= MAX_STUCK_RETRIES:
                                    print(f"[Spot] Stuck navigating to '{name}', retrying ({stuck_retries}/{MAX_STUCK_RETRIES})...")
                                    nav_to_cmd_id = None  # Fresh command on retry
                                    time.sleep(2.0)
                                    continue
                                else:
                                    print(f"[Spot] Still stuck after {MAX_STUCK_RETRIES} retries for '{name}'")
                                    break
                            else:
                                print(f"[Spot] Navigation error for '{name}': {nav_ex}")
                                break

                        try:
                            feedback = graph_nav_client.navigation_feedback(nav_to_cmd_id)
                            if feedback.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
                                print(f"[Spot] Reached '{name}'")
                                waypoint_reached = True
                                break
                            elif feedback.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
                                stuck_retries += 1
                                if stuck_retries <= MAX_STUCK_RETRIES:
                                    print(f"[Spot] Stuck navigating to '{name}', retrying ({stuck_retries}/{MAX_STUCK_RETRIES})...")
                                    nav_to_cmd_id = None  # Fresh command on retry
                                    time.sleep(2.0)
                                    continue
                                else:
                                    print(f"[Spot] Still stuck after {MAX_STUCK_RETRIES} retries for '{name}'")
                                    break
                            elif feedback.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
                                print(f"[Spot] Robot lost during navigation to '{name}'")
                                try:
                                    if self._conn._session and self.ensure_localized(graph_nav_client, self._conn._session):
                                        print(f"[Spot] Re-localized, retrying navigation to '{name}'")
                                        nav_to_cmd_id = None
                                        continue
                                except Exception as e:
                                    print(f"[Spot] Re-localize failed during nav-feedback recovery: "
                                          f"{type(e).__name__}: {e}")
                                break
                            elif feedback.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED:
                                print(f"[Spot] Robot impaired during navigation to '{name}'")
                                return
                            elif feedback.status in (
                                graph_nav_pb2.NavigationFeedbackResponse.STATUS_COMMAND_TIMED_OUT,
                                graph_nav_pb2.NavigationFeedbackResponse.STATUS_CONSTRAINT_FAULT,
                            ):
                                # Command expired or constraint fault — re-send with fresh command
                                nav_to_cmd_id = None
                                continue
                        except Exception as e:
                            # navigation_feedback() can fail mid-traversal on a
                            # GraphNav RPC blip, server restart, etc. The old
                            # silent `except: pass` would let the inner loop
                            # spin forever polling against a dead command — log
                            # and break out so the outer waypoint loop can move
                            # on (or the user can re-issue).
                            print(f"[Spot] navigation_feedback error for '{name}': {e}")
                            break

                        if stop_event.wait(0.5):
                            print("[Spot] Navigation cancelled")
                            return

                    if not waypoint_reached and len(waypoint_ids) > 1:
                        print(f"[Spot] Skipping '{name}', moving to next waypoint...")
                        all_reached = False
                        continue
                    elif not waypoint_reached:
                        print(f"[Spot] Could not reach '{name}'")
                        return False

                if not repeat:
                    print("[Spot] Tour complete — visited all locations!")
                    return all_reached
        except Exception as e:
            print(f"[Spot] Navigation thread error: {e}")
        return False

    def start_thread(self, graph_nav_client, waypoint_ids, location_names, repeat=False):
        """Cancel any existing nav and start a new multi-waypoint navigation thread.
        Returns False (and starts nothing) if a stop arrived since this dispatch began."""
        from src.spot.robot import motion as _motion
        self.cancel()
        gen = self._safety.dispatch_current()
        return self.spawn(
            lambda ev: _motion.thread_then_restore(
                lambda: self.navigate_sequence(
                    graph_nav_client, waypoint_ids, location_names, ev, repeat, gen), ev),
            mode="graphnav", gen=gen)


# ---------------------------------------------------------------- old names
def _nav():
    import src.spot.robot as robot
    return robot.NAV


def is_navigating() -> bool:
    """Check if the robot is currently navigating (motor noise expected)."""
    return _nav().is_navigating()


def is_follow_mode() -> bool:
    """Check if robot is in follow-person mode (lighter motor noise)."""
    return _nav().is_follow_mode()


def _spawn_nav(fn, mode=None, gen=None) -> bool:
    return _nav().spawn(fn, mode=mode, gen=gen)


def _cancel_nav():
    _nav().cancel()


def _ensure_localized(graph_nav_client, session) -> bool:
    return _nav().ensure_localized(graph_nav_client, session)


def _force_relocalize(graph_nav_client, session) -> bool:
    return _nav().force_relocalize(graph_nav_client, session)


def _save_home_waypoint(graph_nav_client):
    _nav().save_home_waypoint(graph_nav_client)


def _navigate_waypoint_sequence(graph_nav_client, waypoint_ids, location_names,
                                stop_event, repeat=False, gen=None):
    return _nav().navigate_sequence(graph_nav_client, waypoint_ids, location_names,
                                    stop_event, repeat, gen)


def _start_nav_thread(graph_nav_client, waypoint_ids, location_names, repeat=False):
    return _nav().start_thread(graph_nav_client, waypoint_ids, location_names, repeat)
