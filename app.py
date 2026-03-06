"""
app.py
------
Gradio entry point for the Natural Language Vehicle Commander (NLVC).
Provides a sleek dark-themed interface with:
  - Left panel: Live CARLA camera feed (streaming)
  - Right panel: Command chatbot + status dashboard
"""

import time
import threading
import gradio as gr
import numpy as np
from engine.carla_client   import CARLAClient
from engine.llm_parser     import parse_command
from engine.vehicle_control import VehicleController

# ── Global State ───────────────────────────────────────────────────────────────
client     = CARLAClient()
controller = None
connected  = False

# ── Connection Bootstrap ───────────────────────────────────────────────────────

def initialize_simulation():
    """Connect to CARLA, spawn vehicle, attach camera, start control loop."""
    global controller, connected

    if not client.connect():
        return "❌ Could not connect to CARLA. Make sure CarlaUE4.exe is running."
    if not client.spawn_vehicle():
        return "❌ Could not spawn vehicle."
    if not client.attach_camera():
        return "❌ Could not attach camera sensor."

    controller = VehicleController(client)
    controller.start()
    connected = True
    return "✅ Connected! Vehicle spawned and camera active. Start driving!"


def shutdown_simulation():
    """Stop controller and clean up all CARLA actors."""
    global connected
    connected = False
    if controller:
        controller.stop()
    client.cleanup()


# ── Camera Feed ────────────────────────────────────────────────────────────────

def get_camera_frame():
    """
    Continuously yield the latest camera frame for the Gradio Image component.
    Falls back to a dark placeholder if no frame is available yet.
    """
    placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
    placeholder[340:380, 560:720] = [40, 40, 40]  # subtle center marker

    while True:
        if connected:
            frame = client.get_latest_frame()
            yield frame if frame is not None else placeholder
        else:
            yield placeholder
        time.sleep(0.05)  # ~20 FPS


# ── Command Handler ────────────────────────────────────────────────────────────

def handle_command(user_message: str, chat_history: list):
    """
    Process a natural language driving command through the full pipeline:
    user text → LLM parser → vehicle controller → CARLA
    """
    if not connected:
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": "⚠️ Not connected to CARLA. Click **Connect** first."})
        return chat_history, ""

    if not user_message.strip():
        return chat_history, ""

    # Parse command through LLM
    intent = parse_command(user_message)
    controller.set_intent(intent)

    # Get current vehicle speed
    speed = client.get_vehicle_speed()

    # Format bot response
    intent_icons = {
        "drive":      "🟢 Driving",
        "stop":       "🔴 Stopping",
        "turn_left":  "↰  Turning Left",
        "turn_right": "↱  Turning Right",
        "reverse":    "🔁 Reversing",
        "unknown":    "❓ Unknown Command",
    }
    icon  = intent_icons.get(intent["intent"], "❓")
    urgency_tag = " ⚡ IMMEDIATE" if intent["urgency"] == "immediate" else ""

    response = (
        f"{icon}{urgency_tag}\n"
        f"```\n"
        f"intent       : {intent['intent']}\n"
        f"speed target : {intent['speed_target']} km/h\n"
        f"urgency      : {intent['urgency']}\n"
        f"current speed: {speed:.1f} km/h\n"
        f"```"
    )

    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": response})
    return chat_history, ""


# ── Status Bar ─────────────────────────────────────────────────────────────────

def get_status():
    """Return a live status string shown below the camera feed."""
    if not connected:
        return "🔴  Disconnected — Launch CARLA and click Connect"
    speed = client.get_vehicle_speed()
    transform = client.get_vehicle_transform()
    if transform:
        loc = transform.location
        return (
            f"🟢  Connected  |  "
            f"Speed: {speed:.1f} km/h  |  "
            f"Position: ({loc.x:.1f}, {loc.y:.1f})"
        )
    return "🟢  Connected"


# ── Custom CSS ─────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600&display=swap');

:root {
    --bg-dark:    #0a0c10;
    --bg-panel:   #111318;
    --bg-card:    #181c24;
    --accent:     #00e5ff;
    --accent-dim: #007a8a;
    --green:      #00ff88;
    --red:        #ff3860;
    --text:       #c8d0e0;
    --text-dim:   #6b7280;
    --border:     #1e2530;
    --font-mono:  'Share Tech Mono', monospace;
    --font-body:  'Barlow', sans-serif;
}

body, .gradio-container {
    background: var(--bg-dark) !important;
    font-family: var(--font-body) !important;
    color: var(--text) !important;
}

/* Header */
#nlvc-header {
    text-align: center;
    padding: 24px 0 8px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
}
#nlvc-header h1 {
    font-family: var(--font-mono) !important;
    font-size: 1.8rem !important;
    color: var(--accent) !important;
    letter-spacing: 0.15em;
    margin: 0;
    text-shadow: 0 0 20px rgba(0, 229, 255, 0.3);
}
#nlvc-header p {
    color: var(--text-dim);
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    margin: 6px 0 0;
}

/* Panels */
.panel-box {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0 !important;
    overflow: hidden;
}

/* Camera feed */
#camera-feed {
    border-radius: 6px !important;
    border: 1px solid var(--accent-dim) !important;
    box-shadow: 0 0 24px rgba(0, 229, 255, 0.08) !important;
}

/* Status bar */
#status-bar textarea, #status-bar input {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    color: var(--green) !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    padding: 8px 12px !important;
}

/* Chatbot */
#command-chat {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: var(--font-body) !important;
}
#command-chat .message.user {
    background: #1a2a3a !important;
    color: var(--accent) !important;
    font-size: 0.9rem !important;
}
#command-chat .message.bot {
    background: var(--bg-panel) !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
}

/* Input box */
#command-input textarea {
    font-family: var(--font-body) !important;
    font-size: 0.95rem !important;
    background: var(--bg-card) !important;
    color: var(--text) !important;
    border: 1px solid var(--accent-dim) !important;
    border-radius: 6px !important;
}
#command-input textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 8px rgba(0, 229, 255, 0.2) !important;
}

/* Buttons */
#connect-btn, #disconnect-btn, #send-btn {
    font-family: var(--font-mono) !important;
    letter-spacing: 0.05em !important;
    border-radius: 4px !important;
    border: none !important;
    font-size: 0.85rem !important;
}
#connect-btn {
    background: var(--accent) !important;
    color: #000 !important;
}
#connect-btn:hover {
    background: var(--green) !important;
}
#disconnect-btn {
    background: transparent !important;
    border: 1px solid var(--red) !important;
    color: var(--red) !important;
}
#send-btn {
    background: var(--accent-dim) !important;
    color: #fff !important;
}
#send-btn:hover {
    background: var(--accent) !important;
    color: #000 !important;
}

/* Section labels */
.section-label {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--accent-dim);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 10px 14px 6px;
    border-bottom: 1px solid var(--border);
}
"""

# ── Build UI ───────────────────────────────────────────────────────────────────

with gr.Blocks(title="NLVC — Natural Language Vehicle Commander") as demo:

    # Header
    gr.HTML("""
        <div id="nlvc-header">
            <h1>◈ NATURAL LANGUAGE VEHICLE COMMANDER</h1>
            <p>NLVC · CARLA Simulator · LLaMA 3 8B · Ollama</p>
        </div>
    """)

    with gr.Row():

        # ── Left Column: Camera + Status ───────────────────────────────────────
        with gr.Column(scale=3):
            gr.HTML('<div class="section-label">◎ Live Camera Feed</div>')
            camera = gr.Image(
                label="",
                streaming=True,
                elem_id="camera-feed",
                show_label=False,
                height=420,
            )
            status = gr.Textbox(
                value="🔴  Disconnected — Launch CARLA and click Connect",
                label="",
                interactive=False,
                elem_id="status-bar",
                show_label=False,
            )
            with gr.Row():
                connect_btn    = gr.Button("[ CONNECT ]",    elem_id="connect-btn")
                disconnect_btn = gr.Button("[ DISCONNECT ]", elem_id="disconnect-btn")

        # ── Right Column: Chatbot + Input ──────────────────────────────────────
        with gr.Column(scale=2):
            gr.HTML('<div class="section-label">◉ Command Interface</div>')
            chatbot = gr.Chatbot(
                value=[],
                label="",
                elem_id="command-chat",
                show_label=False,
                height=360,
            )
            with gr.Row():
                command_input = gr.Textbox(
                    placeholder='Type a command... e.g. "Drive forward at 40 km/h"',
                    label="",
                    elem_id="command-input",
                    show_label=False,
                    scale=4,
                )
                send_btn = gr.Button("SEND", elem_id="send-btn", scale=1)

            gr.HTML("""
                <div style="padding:12px 4px 0; font-size:0.75rem; color:#4b5563; font-family:'Share Tech Mono',monospace; line-height:1.8;">
                    TRY: &nbsp;"Drive at 50 km/h" &nbsp;·&nbsp; "Turn left" &nbsp;·&nbsp; "Hit the brakes!"<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"Speed up to 80" &nbsp;·&nbsp; "Go in reverse" &nbsp;·&nbsp; "Stop"
                </div>
            """)

    # ── Event Wiring ───────────────────────────────────────────────────────────

    connect_btn.click(
        fn=initialize_simulation,
        outputs=status,
    )

    disconnect_btn.click(
        fn=shutdown_simulation,
        outputs=None,
    )

    # Send on button click
    send_btn.click(
        fn=handle_command,
        inputs=[command_input, chatbot],
        outputs=[chatbot, command_input],
    )

    # Send on Enter key
    command_input.submit(
        fn=handle_command,
        inputs=[command_input, chatbot],
        outputs=[chatbot, command_input],
    )

    # Refresh status every 2 seconds
    status_timer = gr.Timer(2)
    status_timer.tick(fn=get_status, outputs=status)

    # Stream camera frames
    demo.load(fn=get_camera_frame, outputs=camera)


# ── Launch ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
            css=CSS,
        )
    finally:
        shutdown_simulation()