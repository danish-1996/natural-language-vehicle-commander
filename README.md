# 🚗 Natural Language Vehicle Commander (NLVC)

> Control a simulated vehicle using plain English — powered by a locally hosted LLM and the CARLA simulator.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![CARLA](https://img.shields.io/badge/CARLA-0.9.14+-orange)
![Ollama](https://img.shields.io/badge/LLM-Ollama%20%7C%20Llama3-purple)
![Gradio](https://img.shields.io/badge/UI-Gradio-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

NLVC is an intelligent middleware layer that bridges the gap between **conversational language** and **real-time vehicle physics**. Type a command like _"slow down and turn left"_ — and watch the simulated car execute it.

The system uses a locally hosted LLM (via Ollama) to parse free-form English into a structured JSON control schema, which is then translated into `carla.VehicleControl()` parameters applied in real time.

This project is fully **privacy-preserving** — no API keys, no cloud calls. Everything runs on your local machine.

---

## 💡 How It Works

```
User: "Hit the brakes right now!"
  → LLM:  {"intent": "stop", "speed_target": 0, "urgency": "immediate"}
  → Car:  carla.VehicleControl(brake=1.0, throttle=0.0)  ✅

User: "Turn left slowly"
  → LLM:  {"intent": "turn_left", "speed_target": 10, "urgency": "normal"}
  → Car:  carla.VehicleControl(throttle=0.08, steer=-0.3)  ✅
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Gradio Web UI                        │
│   [ Live Camera Feed ]     [ Command Chatbox ]           │
└────────────────┬────────────────────────┬────────────────┘
                 │                        │
        Camera Frame Queue        User Text Input
                 │                        │
                 │             ┌──────────▼──────────┐
                 │             │   LLM Inference      │
                 │             │  (Ollama / Llama3)   │
                 │             │  → JSON Intent       │
                 │             └──────────┬──────────┘
                 │                        │
        ┌────────▼────────────────────────▼────────┐
        │           CARLA Control Loop              │
        │   carla.VehicleControl(throttle, steer,   │
        │                        brake)             │
        └───────────────────────────────────────────┘
```

### Components

| Thread | Responsibility |
|---|---|
| **Gradio UI** | Renders live camera feed, accepts user commands, displays command history |
| **LLM Inference** | Injects user input into system prompt, queries Ollama, validates JSON response |
| **CARLA Control Loop** | Reads parsed JSON state, applies vehicle control parameters in real time |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Simulation | [CARLA Simulator](https://carla.org/) 0.9.14+ |
| LLM Backend | [Ollama](https://ollama.com/) serving `llama3:8b` or `mistral` |
| UI Framework | [Gradio](https://gradio.app/) (`gr.Blocks`) |
| Language | Python 3.10+ |
| Version Control | Git (feature-branch workflow) |

---

## 📁 Project Structure

```
natural-language-vehicle-commander/
├── app.py                  # Gradio entry point and UI layout
├── engine/
│   ├── __init__.py
│   ├── llm_parser.py       # Ollama HTTP requests & JSON validation
│   ├── carla_client.py     # CARLA client, vehicle spawn, sensor setup
│   └── vehicle_control.py  # Maps JSON intents → VehicleControl params
├── prompts/
│   └── system_prompt.txt   # Few-shot prompt engineering
├── assets/                 # Architecture diagrams, demo GIFs
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Prerequisites

Before running NLVC, ensure the following are installed and running:

1. **CARLA Simulator** — Download from the [official releases](https://github.com/carla-simulator/carla/releases). Launch the server:
   ```bash
   ./CarlaUE4.sh    # Linux
   CarlaUE4.exe     # Windows
   ```

2. **Ollama** — Install from [ollama.com](https://ollama.com), then pull a model:
   ```bash
   ollama pull llama3:8b
   ```

3. **Python 3.10+** with a virtual environment.

> 💡 **Hardware Note:** A GPU with high VRAM (e.g., RTX 3090 24GB) is recommended to hold LLM weights in memory while simultaneously rendering the CARLA environment without bottlenecking.

---

## 🚀 Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/natural-language-vehicle-commander.git
cd natural-language-vehicle-commander

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Ensure CARLA is running (localhost:2000) and Ollama is serving

# 5. Launch the app
python app.py
```

Open your browser at `http://localhost:7860` to access the Gradio interface.

---

## 💬 Supported Commands (Examples)

| Natural Language Input | Parsed Intent | Action |
|---|---|---|
| `"Drive forward at 30 km/h"` | `drive, 30` | Throttle applied |
| `"Hit the brakes right now!"` | `stop, urgency=immediate` | Full brake |
| `"Turn left slowly"` | `turn_left, speed=10` | Steer + low throttle |
| `"Speed up to 60"` | `drive, 60` | Gradual throttle increase |
| `"Stop the car"` | `stop, speed=0` | Brake applied |

---

## 🧠 Prompt Engineering

The LLM is constrained via a structured system prompt to **always** return a valid JSON object — never conversational text.

```
You are a vehicle command translation engine.
Extract the driving intent and output ONLY valid JSON:
{
  "intent": string,       // "drive" | "stop" | "turn_left" | "turn_right"
  "speed_target": integer, // target speed in km/h
  "urgency": string       // "normal" | "immediate"
}
```

Ambiguous or unrecognized commands map to `"intent": "unknown"`, keeping the vehicle in its current safe state.

---

## 🛡️ Error Handling

| Scenario | Behavior |
|---|---|
| Malformed JSON from LLM | Caught by `JSONDecodeError` → safe brake applied |
| CARLA server offline | `timeout=10s` → UI displays reconnection instructions |
| Ambiguous/gibberish input | LLM returns `"intent": "unknown"` → vehicle holds state |

---

## 🗺️ Development Milestones

- [x] **Milestone 1** — Spawn vehicle, attach RGB camera, save frame to disk
- [ ] **Milestone 2** — Finalize system prompt, validate consistent JSON output from Ollama
- [ ] **Milestone 3** — Connect JSON parser to `vehicle_control.py` (text → movement)
- [ ] **Milestone 4** — Wrap async camera feed + text input into Gradio UI
- [ ] **Milestone 5** — Record demo GIF, finalize README, publish to GitHub

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

[MIT](LICENSE)

---

<p align="center">Built with 🧠 LLMs + 🚗 CARLA + ☕ too much caffeine</p>