"""
carla_client.py
---------------
Manages all interaction with the CARLA simulation environment.
Responsibilities:
  - Connect to the CARLA server (localhost:2000)
  - Spawn an ego vehicle at a random spawn point
  - Attach an RGB camera sensor to the vehicle's hood
  - Push camera frames into a thread-safe queue for the Gradio UI
  - Provide clean teardown of all actors on exit
"""

import queue
import logging
import threading
import numpy as np

try:
    import carla
except ImportError:
    raise ImportError(
        "CARLA Python API not found.\n"
        "Ensure the CARLA Python package is installed or added to your PYTHONPATH.\n"
        "Typically found at: <CARLA_ROOT>/PythonAPI/carla/dist/carla-*.egg"
    )

# ── Configuration ─────────────────────────────────────────────────────────────

CARLA_HOST         = "localhost"
CARLA_PORT         = 2000
CONNECTION_TIMEOUT = 10.0      # seconds to wait for CARLA server

# Vehicle blueprint filter — any car model
VEHICLE_FILTER     = "vehicle.tesla.model3"
VEHICLE_ROLE       = "ego"

# Camera sensor settings
CAMERA_WIDTH       = 1280
CAMERA_HEIGHT      = 720
CAMERA_FOV         = 90        # degrees
CAMERA_TRANSFORM   = carla.Transform(
    carla.Location(x=1.5, z=2.4),   # hood-mounted, slightly elevated
    carla.Rotation(pitch=-10.0),     # angled slightly downward
)

# Frame queue — holds latest camera images for the Gradio UI
# maxsize=1 ensures we always show the latest frame, never a stale one
FRAME_QUEUE_SIZE   = 1

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("carla_client")


# ── CARLAClient Class ──────────────────────────────────────────────────────────

class CARLAClient:
    """
    Manages the lifecycle of a CARLA simulation session.

    Usage:
        client = CARLAClient()
        client.connect()
        client.spawn_vehicle()
        client.attach_camera()

        # In your control loop:
        frame = client.get_latest_frame()
        client.apply_control(throttle=0.5, steer=0.0, brake=0.0)

        # On shutdown:
        client.cleanup()
    """

    def __init__(self):
        self._client   = None
        self._world    = None
        self._vehicle  = None
        self._camera   = None
        self._frame_q  = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self._lock     = threading.Lock()
        self._running  = False

    # ── Connection ─────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """
        Establish connection to the CARLA server.

        Returns:
            True if connected successfully, False otherwise.
        """
        try:
            logger.info(f"Connecting to CARLA at {CARLA_HOST}:{CARLA_PORT} ...")
            self._client = carla.Client(CARLA_HOST, CARLA_PORT)
            self._client.set_timeout(CONNECTION_TIMEOUT)
            self._world = self._client.get_world()
            server_version = self._client.get_server_version()
            logger.info(f"Connected! CARLA server version: {server_version}")
            return True
        except RuntimeError as e:
            logger.error(
                f"Failed to connect to CARLA: {e}\n"
                "Is the CARLA executable running? Launch it first:\n"
                "  Windows: CarlaUE4.exe\n"
                "  Linux:   ./CarlaUE4.sh"
            )
            return False

    # ── Vehicle ────────────────────────────────────────────────────────────────

    def spawn_vehicle(self) -> bool:
        """
        Spawn the ego vehicle at a random spawn point in the current map.

        Returns:
            True if spawned successfully, False otherwise.
        """
        if not self._world:
            logger.error("Not connected to CARLA. Call connect() first.")
            return False

        blueprint_library = self._world.get_blueprint_library()

        # Get vehicle blueprint
        vehicle_bp = blueprint_library.find(VEHICLE_FILTER)
        if vehicle_bp is None:
            logger.warning(
                f"Blueprint '{VEHICLE_FILTER}' not found. Falling back to random vehicle."
            )
            vehicle_bp = blueprint_library.filter("vehicle.*")[0]

        vehicle_bp.set_attribute("role_name", VEHICLE_ROLE)

        # Get a random valid spawn point
        spawn_points = self._world.get_map().get_spawn_points()
        if not spawn_points:
            logger.error("No spawn points available in the current map.")
            return False

        import random
        spawn_transform = random.choice(spawn_points)

        try:
            self._vehicle = self._world.spawn_actor(vehicle_bp, spawn_transform)
            logger.info(
                f"Spawned vehicle: {self._vehicle.type_id} "
                f"at {spawn_transform.location}"
            )
            self._running = True
            return True
        except RuntimeError as e:
            logger.error(f"Failed to spawn vehicle: {e}")
            return False

    # ── Camera Sensor ──────────────────────────────────────────────────────────

    def attach_camera(self) -> bool:
        """
        Attach an RGB camera sensor to the ego vehicle.
        Camera frames are pushed into the internal frame queue automatically.

        Returns:
            True if attached successfully, False otherwise.
        """
        if not self._vehicle:
            logger.error("No vehicle spawned. Call spawn_vehicle() first.")
            return False

        blueprint_library = self._world.get_blueprint_library()
        camera_bp = blueprint_library.find("sensor.camera.rgb")

        # Set camera resolution and FOV
        camera_bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
        camera_bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
        camera_bp.set_attribute("fov",           str(CAMERA_FOV))

        # Attach to vehicle
        self._camera = self._world.spawn_actor(
            camera_bp,
            CAMERA_TRANSFORM,
            attach_to=self._vehicle
        )

        # Register callback — fires every time a new frame is rendered
        self._camera.listen(self._on_camera_frame)
        logger.info(
            f"Camera attached at {CAMERA_TRANSFORM.location} | "
            f"{CAMERA_WIDTH}x{CAMERA_HEIGHT} @ FOV {CAMERA_FOV}°"
        )
        return True

    def _on_camera_frame(self, image: "carla.Image") -> None:
        """
        Callback fired by CARLA on every new camera frame.
        Converts raw BGRA bytes → RGB numpy array and pushes to the frame queue.
        Drops the oldest frame if the queue is full (keeps UI fresh).
        """
        # CARLA images are BGRA — convert to RGB for display
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        rgb   = array[:, :, :3][:, :, ::-1].copy()  # BGRA → RGB

        # Non-blocking put — drop old frame if queue is full
        try:
            self._frame_q.put_nowait(rgb)
        except queue.Full:
            try:
                self._frame_q.get_nowait()   # discard stale frame
            except queue.Empty:
                pass
            self._frame_q.put_nowait(rgb)

    def get_latest_frame(self) -> np.ndarray | None:
        """
        Retrieve the most recent camera frame without blocking.

        Returns:
            RGB numpy array (H, W, 3) or None if no frame is available yet.
        """
        try:
            return self._frame_q.get_nowait()
        except queue.Empty:
            return None

    # ── Vehicle Control ────────────────────────────────────────────────────────

    def apply_control(
        self,
        throttle: float = 0.0,
        steer:    float = 0.0,
        brake:    float = 0.0,
        reverse:  bool  = False,
    ) -> None:
        """
        Apply VehicleControl parameters to the ego vehicle.

        Args:
            throttle: 0.0 – 1.0
            steer:   -1.0 (full left) to +1.0 (full right)
            brake:    0.0 – 1.0
            reverse:  True to engage reverse gear
        """
        if not self._vehicle:
            logger.warning("apply_control() called but no vehicle is spawned.")
            return

        control = carla.VehicleControl(
            throttle = float(np.clip(throttle, 0.0, 1.0)),
            steer    = float(np.clip(steer,   -1.0, 1.0)),
            brake    = float(np.clip(brake,    0.0, 1.0)),
            reverse  = reverse,
        )
        self._vehicle.apply_control(control)

    def emergency_stop(self) -> None:
        """Immediately apply full brakes — used as a safety fallback."""
        logger.warning("Emergency stop triggered!")
        self.apply_control(throttle=0.0, steer=0.0, brake=1.0)

    # ── World Utilities ────────────────────────────────────────────────────────

    def get_vehicle_speed(self) -> float:
        """
        Returns current vehicle speed in km/h.
        """
        if not self._vehicle:
            return 0.0
        velocity  = self._vehicle.get_velocity()
        speed_ms  = (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5
        return speed_ms * 3.6  # m/s → km/h

    def get_vehicle_transform(self):
        """Returns the vehicle's current carla.Transform (location + rotation)."""
        if not self._vehicle:
            return None
        return self._vehicle.get_transform()

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """
        Destroy all spawned actors (camera + vehicle) and reset state.
        Always call this on shutdown to avoid ghost actors in CARLA.
        """
        logger.info("Cleaning up CARLA actors...")
        self._running = False

        if self._camera and self._camera.is_alive:
            self._camera.stop()
            self._camera.destroy()
            logger.info("Camera destroyed.")

        if self._vehicle and self._vehicle.is_alive:
            self.emergency_stop()
            self._vehicle.destroy()
            logger.info("Vehicle destroyed.")

        self._camera  = None
        self._vehicle = None
        self._world   = None
        self._client  = None
        logger.info("Cleanup complete.")

    # ── Context Manager Support ────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# ── Quick test harness ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    logger.info("Starting CARLA client test...")

    with CARLAClient() as client:
        # Step 1: Connect
        if not client.connect():
            logger.error("Aborting — could not connect to CARLA.")
            exit(1)

        # Step 2: Spawn vehicle
        if not client.spawn_vehicle():
            logger.error("Aborting — could not spawn vehicle.")
            exit(1)

        # Step 3: Attach camera
        if not client.attach_camera():
            logger.error("Aborting — could not attach camera.")
            exit(1)

        logger.info("Waiting for first camera frame...")
        time.sleep(2)  # give CARLA time to render the first frame

        # Step 4: Grab and save a frame
        frame = client.get_latest_frame()
        if frame is not None:
            from PIL import Image
            img = Image.fromarray(frame)
            img.save("test_frame.png")
            logger.info(f"Frame saved to test_frame.png | Shape: {frame.shape}")
        else:
            logger.warning("No frame received yet.")

        # Step 5: Test basic controls
        logger.info("Testing throttle for 3 seconds...")
        client.apply_control(throttle=0.5, steer=0.0)
        time.sleep(3)

        speed = client.get_vehicle_speed()
        logger.info(f"Current speed: {speed:.1f} km/h")

        logger.info("Applying brakes...")
        client.apply_control(brake=1.0)
        time.sleep(2)

        logger.info("Test complete — cleaning up.")