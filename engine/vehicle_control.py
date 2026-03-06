"""
vehicle_control.py
------------------
Translates structured JSON intents from the LLM parser into
real-time carla.VehicleControl() parameters applied to the ego vehicle.

Responsibilities:
  - Map intent + speed_target + urgency → throttle / steer / brake values
  - Implement smooth acceleration (gradual ramp-up, not instant full throttle)
  - Handle all 6 intent types: drive, stop, turn_left, turn_right, reverse, unknown
  - Run a continuous control loop in a background thread
  - Expose a simple set_intent() interface for the Gradio UI
"""

import time
import logging
import threading
from dataclasses import dataclass, field

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vehicle_control")


# ── Control Parameters ─────────────────────────────────────────────────────────

# How often the control loop ticks (seconds)
CONTROL_LOOP_HZ     = 20          # 20 updates per second
CONTROL_INTERVAL    = 1.0 / CONTROL_LOOP_HZ

# Throttle ramp speed — how much throttle increases per tick (smooth acceleration)
THROTTLE_RAMP_RATE  = 0.05        # 0→1 in ~1 second
BRAKE_RAMP_RATE     = 0.15        # brakes apply faster than throttle

# Steering values per intent
STEER_TURN          = 0.4         # moderate turn — not too sharp
STEER_STRAIGHT      = 0.0

# Maximum throttle cap (prevents uncontrolled high speeds)
MAX_THROTTLE        = 0.8

# Speed → throttle mapping table (km/h : base throttle)
SPEED_THROTTLE_MAP  = {
    0:   0.0,
    10:  0.25,
    20:  0.35,
    30:  0.45,
    40:  0.55,
    60:  0.65,
    80:  0.72,
    100: 0.78,
    120: MAX_THROTTLE,
}


# ── Intent State Dataclass ─────────────────────────────────────────────────────

@dataclass
class IntentState:
    """Holds the current driving intent parsed from the LLM."""
    intent:       str   = "stop"
    speed_target: int   = 0
    urgency:      str   = "normal"


# ── VehicleController Class ────────────────────────────────────────────────────

class VehicleController:
    """
    Translates LLM intents into CARLA vehicle control commands.

    Usage:
        controller = VehicleController(carla_client)
        controller.start()

        # When LLM parses a new command:
        controller.set_intent({"intent": "drive", "speed_target": 40, "urgency": "normal"})

        # On shutdown:
        controller.stop()
    """

    def __init__(self, carla_client):
        """
        Args:
            carla_client: An active CARLAClient instance with a spawned vehicle.
        """
        self._client        = carla_client
        self._intent        = IntentState()
        self._lock          = threading.Lock()
        self._thread        = None
        self._running       = False
        self._current_throttle = 0.0
        self._current_brake    = 0.0

    # ── Public Interface ───────────────────────────────────────────────────────

    def set_intent(self, parsed_json: dict) -> None:
        """
        Update the current driving intent from a parsed LLM JSON dict.

        Args:
            parsed_json: Output from llm_parser.parse_command()
                         e.g. {"intent": "drive", "speed_target": 40, "urgency": "normal"}
        """
        with self._lock:
            self._intent = IntentState(
                intent       = parsed_json.get("intent",       "stop"),
                speed_target = parsed_json.get("speed_target", 0),
                urgency      = parsed_json.get("urgency",      "normal"),
            )
        logger.info(
            f"Intent updated → intent={self._intent.intent} | "
            f"speed={self._intent.speed_target} km/h | "
            f"urgency={self._intent.urgency}"
        )

    def start(self) -> None:
        """Start the background control loop thread."""
        if self._running:
            logger.warning("Control loop is already running.")
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._control_loop,
            daemon=True,
            name="VehicleControlLoop"
        )
        self._thread.start()
        logger.info("Vehicle control loop started.")

    def stop(self) -> None:
        """Stop the control loop and apply emergency brakes."""
        logger.info("Stopping vehicle control loop...")
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._client.emergency_stop()
        logger.info("Control loop stopped.")

    # ── Core Control Loop ──────────────────────────────────────────────────────

    def _control_loop(self) -> None:
        """
        Background thread: reads the current intent and applies
        vehicle control parameters at CONTROL_LOOP_HZ frequency.
        """
        while self._running:
            start = time.monotonic()

            # Snapshot the current intent (thread-safe)
            with self._lock:
                intent       = self._intent.intent
                speed_target = self._intent.speed_target
                urgency      = self._intent.urgency

            # Compute and apply control
            throttle, steer, brake, reverse = self._compute_control(
                intent, speed_target, urgency
            )
            self._client.apply_control(
                throttle=throttle,
                steer=steer,
                brake=brake,
                reverse=reverse,
            )

            # Tick at fixed rate
            elapsed = time.monotonic() - start
            sleep_time = CONTROL_INTERVAL - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _compute_control(
        self,
        intent:       str,
        speed_target: int,
        urgency:      str,
    ) -> tuple[float, float, float, bool]:
        """
        Map an intent + speed_target + urgency → (throttle, steer, brake, reverse).

        Returns:
            Tuple of (throttle, steer, brake, reverse)
        """
        reverse = False

        # ── STOP ──────────────────────────────────────────────────────────────
        if intent == "stop":
            brake_value = 1.0 if urgency == "immediate" else self._ramp_brake()
            self._current_throttle = 0.0
            self._current_brake    = brake_value
            return 0.0, STEER_STRAIGHT, brake_value, False

        # ── UNKNOWN — hold current state (do nothing) ──────────────────────────
        if intent == "unknown":
            return 0.0, STEER_STRAIGHT, 0.0, False

        # ── REVERSE ───────────────────────────────────────────────────────────
        if intent == "reverse":
            target_throttle = self._speed_to_throttle(speed_target or 10)
            throttle = self._ramp_throttle(target_throttle, urgency)
            self._current_brake = 0.0
            return throttle, STEER_STRAIGHT, 0.0, True

        # ── DRIVE ─────────────────────────────────────────────────────────────
        if intent == "drive":
            target_throttle = self._speed_to_throttle(speed_target)
            throttle = self._ramp_throttle(target_throttle, urgency)
            self._current_brake = 0.0
            return throttle, STEER_STRAIGHT, 0.0, False

        # ── TURN LEFT ─────────────────────────────────────────────────────────
        if intent == "turn_left":
            target_throttle = self._speed_to_throttle(speed_target or 20)
            throttle = self._ramp_throttle(target_throttle, urgency)
            self._current_brake = 0.0
            return throttle, -STEER_TURN, 0.0, False

        # ── TURN RIGHT ────────────────────────────────────────────────────────
        if intent == "turn_right":
            target_throttle = self._speed_to_throttle(speed_target or 20)
            throttle = self._ramp_throttle(target_throttle, urgency)
            self._current_brake = 0.0
            return throttle, +STEER_TURN, 0.0, False

        # Fallback — unknown intent defaults to safe stop
        logger.warning(f"Unhandled intent '{intent}' — defaulting to stop.")
        return 0.0, STEER_STRAIGHT, 1.0, False

    # ── Helper: Speed → Throttle ───────────────────────────────────────────────

    def _speed_to_throttle(self, speed_kmh: int) -> float:
        """
        Convert a target speed in km/h to a base throttle value
        using the SPEED_THROTTLE_MAP lookup table with linear interpolation.
        """
        if speed_kmh <= 0:
            return 0.0
        if speed_kmh >= 120:
            return MAX_THROTTLE

        # Find surrounding entries in the map
        speeds  = sorted(SPEED_THROTTLE_MAP.keys())
        for i in range(len(speeds) - 1):
            lo, hi = speeds[i], speeds[i + 1]
            if lo <= speed_kmh <= hi:
                # Linear interpolation
                t = (speed_kmh - lo) / (hi - lo)
                return SPEED_THROTTLE_MAP[lo] + t * (
                    SPEED_THROTTLE_MAP[hi] - SPEED_THROTTLE_MAP[lo]
                )
        return MAX_THROTTLE

    # ── Helper: Smooth Ramps ───────────────────────────────────────────────────

    def _ramp_throttle(self, target: float, urgency: str) -> float:
        """
        Gradually ramp throttle toward target (smooth acceleration).
        Immediate urgency jumps directly to target.
        """
        if urgency == "immediate":
            self._current_throttle = target
        else:
            if self._current_throttle < target:
                self._current_throttle = min(
                    self._current_throttle + THROTTLE_RAMP_RATE, target
                )
            else:
                self._current_throttle = max(
                    self._current_throttle - THROTTLE_RAMP_RATE, target
                )
        return round(self._current_throttle, 3)

    def _ramp_brake(self) -> float:
        """Gradually ramp brake toward 1.0 for smooth stops."""
        self._current_brake = min(self._current_brake + BRAKE_RAMP_RATE, 1.0)
        return round(self._current_brake, 3)


# ── Quick test harness ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from engine.carla_client import CARLAClient
    from engine.llm_parser   import parse_command

    test_commands = [
        "Drive forward at 40 km/h",
        "Turn left slowly",
        "Speed up to 80",
        "Hit the brakes right now!",
    ]

    with CARLAClient() as client:
        if not client.connect():
            exit(1)
        if not client.spawn_vehicle():
            exit(1)
        if not client.attach_camera():
            exit(1)

        controller = VehicleController(client)
        controller.start()

        for cmd in test_commands:
            logger.info(f"\n{'='*50}")
            logger.info(f"Command: '{cmd}'")
            intent = parse_command(cmd)
            controller.set_intent(intent)

            # Hold each command for 4 seconds
            for _ in range(4):
                time.sleep(1)
                speed = client.get_vehicle_speed()
                logger.info(f"  Speed: {speed:.1f} km/h | Intent: {intent['intent']}")

        controller.stop()
        logger.info("Test complete.")