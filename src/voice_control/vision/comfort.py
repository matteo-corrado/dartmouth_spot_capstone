"""Friendly Spot comfort model, ported unchanged from friendly_spot/src/behavior/planner.py.

Original authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado (COSC 69.15, 25F).
Score = base + action + pose + gesture + emotion + distance + familiarity, mapped
to a behaviour by threshold. Weights and thresholds are the original values.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple


class BehaviorLabel(str, Enum):
    """Canonical Spot behaviors inferred from comfort score."""

    GO_CLOSE = "go_close"
    GO_CLOSE_SLOWLY = "go_close_slowly"
    BACK_AWAY = "back_away"
    BACK_AWAY_SLOWLY = "back_away_slowly"
    STAY = "stay"
    SIT = "sit"


@dataclass
class PerceptionInput:
    """
    Aggregated perception signals used by the comfort model.

    Attributes:
        current_action: High-level Spot action (e.g., "moving", "idle").
        distance_m: Estimated distance between Spot and the human in meters.
        face_label: Name returned by face recognition or "unknown".
        emotion_label: Emotion string emitted by DeepFace (e.g., "happy").
        pose_label: Pose/action classification (e.g., "running", "standing").
        gesture_label: Hand gesture label (e.g., "thumbs_up", "open_hand").
        face_bbox: Face bounding box as (x, y, w, h) tuple, or None if no face detected.
        pose_landmarks: Normalized pose landmarks as numpy array of shape (N, 3), or None.
        emotion_scores: Full emotion score dict from DeepFace, or None.
        frame: The actual frame that was analyzed (BGR image), for visualization alignment.
    """

    current_action: str = "idle"
    distance_m: Optional[float] = None
    face_label: str = "unknown"
    emotion_label: str = "neutral"
    pose_label: str = "standing"
    gesture_label: str = "none"
    face_bbox: Optional[tuple] = None
    pose_landmarks: Optional[object] = None  # numpy array
    emotion_scores: Optional[dict] = None
    frame: Optional[object] = None  # numpy array
    ptz_pan: Optional[float] = None  # PTZ pan angle in degrees
    ptz_tilt: Optional[float] = None  # PTZ tilt angle in degrees


@dataclass
class ComfortHyperParameters:
    """
    Tunable hyperparameters that shape how the comfort score is computed.
    We set reasonable defaults here, but these can be adjusted via the ComfortTuner.
    """

    base_score: float = 0.5
    preferred_distance_m: float = 1.2
    distance_tolerance_m: float = 0.35
    distance_weight: float = 0.18
    familiarity_bonus: float = 0.12
    unfamiliar_penalty: float = -0.18
    calm_down_bonus: float = 0.06
    action_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "moving": -0.1,
            "waiting": 0.03,
            "interacting": 0.05,
            "searching": -0.02,
            "idle": 0.0,
            "sit": 0.08,
            "stopped": 0.1,
            "default": 0.0,
        }
    )
    pose_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "running": -0.35,
            "walking": -0.08,
            "standing": 0.05,
            "arms_crossed": -0.01,
            "waving": 0.12,
            "raising_hand": 0.08,
            "clapping": 0.06,
            "bending": -0.05,
            "unknown": 0.0,
            "default": 0.0,
        }
    )
    gesture_weights: Dict[str, float] = field(
        default_factory=lambda: {
            # Positive/welcoming gestures
            "thumb_up": 0.2,          # MediaPipe: Thumb_Up
            "iloveyou": 0.4,         # MediaPipe: ILoveYou (strong positive)
            "victory": 0.06,          # MediaPipe: Victory (peace sign)
            "open_palm": 0.05,        # MediaPipe: Open_Palm (neutral/welcoming)
            # Negative/cautious gestures
            "thumb_down": -0.25,       # MediaPipe: Thumb_Down (negative feedback)
            "closed_fist": -0.15,     # MediaPipe: Closed_Fist (aggressive)
            "pointing_up": -0.05,     # MediaPipe: Pointing_Up (directive/commanding)
            # Neutral states
            "none": 0.0,              # MediaPipe: None (hands visible, no gesture)
            "unknown": 0.0,           # MediaPipe: Unknown (unrecognized gesture)
            "default": 0.0,           # Fallback for unlabeled gestures
        }
    )
    emotion_weights: Dict[str, float] = field(
        default_factory=lambda: {
            # Positive emotions (DeepFace recognized)
            "happy": 0.4,           # DeepFace: happy - strong positive signal
            "surprise": 0.06,        # DeepFace: surprise - mild positive (neutral-to-positive)
            # Neutral
            "neutral": 0.1,          # DeepFace: neutral - baseline
            # Negative emotions (DeepFace recognized)
            "sad": -0.15,            # DeepFace: sad - moderate negative
            "disgust": -0.22,        # DeepFace: disgust - strong negative
            "fear": -0.27,           # DeepFace: fear - very strong negative
            "angry": -0.4,           # DeepFace: angry - strongest negative
            # Fallback
            "default": 0.0,          # Unknown emotions default to neutral
        }
    )
    # Behavior thresholds are ordered from highest comfort to lowest.
    behavior_thresholds: Tuple[Tuple[float, BehaviorLabel], ...] = (
        (0.85, BehaviorLabel.GO_CLOSE),
        (0.71, BehaviorLabel.GO_CLOSE_SLOWLY),
        (0.6, BehaviorLabel.STAY),
        (0.4, BehaviorLabel.SIT),
        (0.23, BehaviorLabel.BACK_AWAY_SLOWLY),
        (0.0, BehaviorLabel.BACK_AWAY),
    )


class ComfortModel:
    """
    Converts perception outputs into a comfort score and a behavior label.
    """

    def __init__(self, hyperparameters: Optional[ComfortHyperParameters] = None):
        self.hp = hyperparameters or ComfortHyperParameters()

    def score(self, perception: PerceptionInput) -> float:
        """
        Compute a comfort score in the [0, 1] range.
        """

        score = self.hp.base_score
        score += self._lookup(perception.current_action, self.hp.action_weights)
        score += self._lookup(perception.pose_label, self.hp.pose_weights)
        score += self._score_gesture(perception.gesture_label, self.hp.gesture_weights)
        score += self._lookup(perception.emotion_label, self.hp.emotion_weights)
        score += self._score_distance(perception.distance_m)
        score += self._score_familiarity(perception.face_label)

        return self._clamp(score, 0.0, 1.0)

    def predict_behavior(self, perception: PerceptionInput) -> Tuple[float, BehaviorLabel]:
        """
        Calculate comfort and map it into a discrete Spot behavior command.
        """

        comfort = self.score(perception)

        for threshold, behavior in self.hp.behavior_thresholds:
            if comfort >= threshold:
                return comfort, behavior

        # Should never happen because final threshold is 0.0.
        return comfort, BehaviorLabel.BACK_AWAY

    def _score_distance(self, distance_m: Optional[float]) -> float:
        """
        Encourage keeping a preferred distance window. Being too close is treated
        as a stronger negative signal than being too far away.
        """

        if distance_m is None or distance_m < 0:
            return 0.0

        pref = self.hp.preferred_distance_m
        tol = self.hp.distance_tolerance_m

        if distance_m < pref - tol:
            # Spot is closer than desired; apply a penalty.
            overflow = (pref - tol) - distance_m
            penalty = min(1.0, overflow / max(pref, 0.1))
            return -penalty * self.hp.distance_weight

        if distance_m > pref + tol:
            # Spot is farther away; apply a softer penalty to nudge an approach.
            overflow = distance_m - (pref + tol)
            penalty = min(1.0, overflow / max(pref + tol, 0.1))
            return -0.6 * penalty * self.hp.distance_weight

        # Within the comfort band; provide a small boost.
        return 0.5 * self.hp.distance_weight

    def _score_familiarity(self, face_label: Optional[str]) -> float:
        """
        Favor familiar faces and penalize unknown ones.
        """

        if not face_label or face_label.lower() == "unknown":
            return self.hp.unfamiliar_penalty

        return self.hp.familiarity_bonus

    def _score_gesture(self, gesture_label: Optional[str], gesture_weights: Dict[str, float]) -> float:
        """
        Score gestures with special handling for bilateral identical gestures.
        If both hands show the same gesture (e.g., "Left: Thumb_Up, Right: Thumb_Up"),
        double the comfort contribution.
        """
        if not gesture_label:
            return gesture_weights.get("default", 0.0)

        # Check if it's a bilateral gesture (both hands with same gesture)
        if ", " in gesture_label:
            parts = gesture_label.split(", ")
            if len(parts) == 2:
                # Extract gesture names from "Left: Gesture" and "Right: Gesture"
                left_parts = parts[0].split(": ")
                right_parts = parts[1].split(": ")
                
                if len(left_parts) == 2 and len(right_parts) == 2:
                    left_gesture = left_parts[1].lower().replace(" ", "_")
                    right_gesture = right_parts[1].lower().replace(" ", "_")
                    
                    # If both hands show same gesture, double the score
                    if left_gesture == right_gesture:
                        base_score = gesture_weights.get(left_gesture, gesture_weights.get("default", 0.0))
                        return base_score * 2.0
                    else:
                        # Different gestures on each hand - use the lowest (most negative) score
                        left_score = gesture_weights.get(left_gesture, gesture_weights.get("default", 0.0))
                        right_score = gesture_weights.get(right_gesture, gesture_weights.get("default", 0.0))
                        return min(left_score, right_score)

        # Single hand gesture - extract gesture name from "Hand: Gesture" format
        gesture_lower = gesture_label.lower()
        if ": " in gesture_lower:
            parts = gesture_lower.split(": ")
            if len(parts) == 2:
                gesture_name = parts[1].replace(" ", "_")
                return gesture_weights.get(gesture_name, gesture_weights.get("default", 0.0))

        # Fallback to standard lookup for other formats
        return gesture_weights.get(gesture_lower, gesture_weights.get("default", 0.0))

    @staticmethod
    def _lookup(label: Optional[str], table: Dict[str, float]) -> float:
        if not label:
            return table.get("default", 0.0)

        return table.get(label.lower(), table.get("default", 0.0))

    @staticmethod
    def _clamp(value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(value, max_value))


TARGET_COMFORT: Dict[BehaviorLabel, float] = {
    BehaviorLabel.GO_CLOSE: 0.85,
    BehaviorLabel.GO_CLOSE_SLOWLY: 0.7,
    BehaviorLabel.STAY: 0.6,
    BehaviorLabel.SIT: 0.46,
    BehaviorLabel.BACK_AWAY_SLOWLY: 0.23,
    BehaviorLabel.BACK_AWAY: 0.0,
}


class ComfortTuner:
    """
    Lightweight online adaptation helper that nudges hyperparameters whenever a
    human confirms that the selected behavior was correct (or incorrect).

    The tuner treats each table entry (pose weight, gesture weight, etc.) like a
    learnable scalar feature weight. When feedback is positive we push the score
    toward the center of the target behavior's comfort band, reinforcing that
    pattern. When feedback is negative we move the score away from the target
    band, making it less likely the same perception input leads to that behavior.
    """

    def __init__(
        self,
        model: ComfortModel,
        learning_rate: float = 0.05,
        max_abs_weight: float = 0.6,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.max_abs_weight = max_abs_weight

    def register_feedback(
        self,
        perception: PerceptionInput,
        behavior: BehaviorLabel,
        correct: bool = True,
    ) -> None:
        """
        Update hyperparameters so that `perception` maps closer to the comfort
        level associated with `behavior`.
        """

        current_comfort = self.model.score(perception)
        target = TARGET_COMFORT[behavior]

        # If feedback is negative, push in the opposite direction.
        error = (target - current_comfort) * (1 if correct else -1)

        self._update_base_score(error)
        self._update_weight(self.model.hp.action_weights, perception.current_action, error)
        self._update_weight(self.model.hp.pose_weights, perception.pose_label, error)
        self._update_weight(self.model.hp.gesture_weights, perception.gesture_label, error)
        self._update_weight(self.model.hp.emotion_weights, perception.emotion_label, error)
        self._update_familiarity(perception.face_label, error)

    def _update_base_score(self, error: float) -> None:
        delta = self.learning_rate * error
        new_score = ComfortModel._clamp(self.model.hp.base_score + delta, 0.0, 1.0)
        self.model.hp.base_score = new_score

    def _update_weight(self, table: Dict[str, float], label: Optional[str], error: float) -> None:
        if not label:
            label = "default"
        key = label.lower()
        if key not in table:
            table[key] = 0.0
        delta = self.learning_rate * error
        table[key] = self._clip(table[key] + delta)

    def _update_familiarity(self, face_label: Optional[str], error: float) -> None:
        delta = self.learning_rate * error
        if face_label and face_label.lower() != "unknown":
            self.model.hp.familiarity_bonus = self._clip(
                self.model.hp.familiarity_bonus + delta
            )
        else:
            self.model.hp.unfamiliar_penalty = self._clip(
                self.model.hp.unfamiliar_penalty - delta
            )

    def _clip(self, value: float) -> float:
        return max(-self.max_abs_weight, min(self.max_abs_weight, value))


