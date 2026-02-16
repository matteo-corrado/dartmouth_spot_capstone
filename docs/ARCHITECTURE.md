# Spot Voice Control System Architecture

## Status Legend
- **GREEN** = Completed & Working
- **YELLOW** = In Progress / Partial
- **RED/GRAY** = Not Started

---

## System Diagram (Mermaid)

```mermaid
flowchart TB
    subgraph HARDWARE["HARDWARE LAYER"]
        MIC["Microphone\n(16kHz Audio)"]
        JETSON["Jetson AGX Orin\n(64GB)"]
        SPOT["Boston Dynamics\nSpot Robot"]
    end

    subgraph BACKEND["BACKEND (Jetson)"]
        subgraph VOICE["Voice Pipeline"]
            VAD["Voice Activity\nDetection\n(WebRTC VAD)"]
            NR["Noise\nReduction"]
            ASR["ASR Server\n(Faster Whisper)"]
        end

        subgraph PROCESSING["Command Processing"]
            INTENT["Intent Parser\n(Regex Patterns)"]
            DISPATCH["Spot Dispatcher\n(SDK Commands)"]
        end

        subgraph SAFETY["Safety Layer"]
            ESTOP["E-Stop Service\n(Always Running)"]
        end

        subgraph NAV["Navigation"]
            GRAPHNAV["GraphNav Client\n(Waypoint Navigation)"]
            LOCMGR["Location Manager\n(Save/Load Waypoints)"]
        end
    end

    subgraph FRONTEND["SPOT CAPABILITIES (What Users See)"]
        subgraph MOVE_COMPLETE["Movement - COMPLETE"]
            M1["Walk Forward/Back"]
            M2["Strafe Left/Right"]
            M3["Turn Left/Right"]
            M4["Turn Around"]
        end

        subgraph POSTURE_COMPLETE["Posture - COMPLETE"]
            P1["Stand / Sit"]
            P2["Crouch / Stand Tall"]
            P3["Self-Right Recovery"]
        end

        subgraph NAV_COMPLETE["Navigation - COMPLETE"]
            N1["Go To Location"]
            N2["Save Location"]
            N3["List Locations"]
        end

        subgraph STATUS_COMPLETE["Status - COMPLETE"]
            S1["Battery Status"]
            S2["Robot Status"]
            S3["Power Off"]
        end

        subgraph SAFETY_COMPLETE["Safety - COMPLETE"]
            SF1["Stop / Halt"]
            SF2["Freeze"]
            SF3["Emergency Stop"]
        end

        subgraph NOT_STARTED["Not Started"]
            NS1["Person Following"]
            NS2["PTZ Camera Control"]
            NS3["LLM Natural Language"]
        end
    end

    %% Connections
    MIC --> VAD
    VAD --> NR
    NR --> ASR
    ASR --> INTENT
    INTENT --> DISPATCH
    DISPATCH --> SPOT
    DISPATCH --> GRAPHNAV
    GRAPHNAV --> LOCMGR
    ESTOP -.->|"Safety Override"| SPOT
    JETSON -.->|"Hosts"| BACKEND

    %% Styling
    classDef complete fill:#22c55e,stroke:#16a34a,color:#fff
    classDef inprogress fill:#eab308,stroke:#ca8a04,color:#000
    classDef notstarted fill:#6b7280,stroke:#4b5563,color:#fff
    classDef hardware fill:#3b82f6,stroke:#2563eb,color:#fff

    class M1,M2,M3,M4,P1,P2,P3,N1,N2,N3,S1,S2,S3,SF1,SF2,SF3 complete
    class NS1,NS2,NS3 notstarted
    class MIC,JETSON,SPOT hardware
    class VAD,NR,ASR,INTENT,DISPATCH,ESTOP,GRAPHNAV,LOCMGR complete
```

---

## System Diagram (ASCII)

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         SPOT VOICE CONTROL SYSTEM                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                        HARDWARE LAYER                                   │ ║
║  │  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────┐    │ ║
║  │  │  Microphone  │    │  Jetson AGX Orin │    │   Boston Dynamics  │    │ ║
║  │  │  (16kHz)     │    │  (64GB RAM)      │    │      Spot Robot    │    │ ║
║  │  └──────┬───────┘    └────────┬─────────┘    └─────────▲──────────┘    │ ║
║  └─────────┼─────────────────────┼─────────────────────────┼──────────────┘ ║
║            │                     │                         │                 ║
║            ▼                     │ HOSTS                   │                 ║
║  ┌─────────────────────────────────────────────────────────┼───────────────┐ ║
║  │                      BACKEND (Jetson)                   │               │ ║
║  │                                                         │               │ ║
║  │  ╔═══════════════════════════════════════════════════╗  │               │ ║
║  │  ║            VOICE PIPELINE [COMPLETE]              ║  │               │ ║
║  │  ║  ┌─────────┐   ┌─────────┐   ┌─────────────────┐  ║  │               │ ║
║  │  ║  │   VAD   │──▶│  Noise  │──▶│   ASR Server    │  ║  │               │ ║
║  │  ║  │(WebRTC) │   │Reduction│   │(Faster Whisper) │  ║  │               │ ║
║  │  ║  └─────────┘   └─────────┘   └────────┬────────┘  ║  │               │ ║
║  │  ╚═══════════════════════════════════════┼═══════════╝  │               │ ║
║  │                                          │              │               │ ║
║  │                                          ▼              │               │ ║
║  │  ╔═══════════════════════════════════════════════════╗  │               │ ║
║  │  ║         COMMAND PROCESSING [COMPLETE]             ║  │               │ ║
║  │  ║  ┌─────────────────┐   ┌────────────────────────┐ ║  │               │ ║
║  │  ║  │  Intent Parser  │──▶│    Spot Dispatcher     │─╫──┼───────────────┤ ║
║  │  ║  │ (Regex Patterns)│   │    (SDK Commands)      │ ║  │               │ ║
║  │  ║  └─────────────────┘   └───────────┬────────────┘ ║  │               │ ║
║  │  ╚════════════════════════════════════┼══════════════╝  │               │ ║
║  │                                       │                 │               │ ║
║  │                                       ▼                 │               │ ║
║  │  ╔═════════════════════════════════════════════════╗    │               │ ║
║  │  ║          NAVIGATION [COMPLETE]                  ║    │               │ ║
║  │  ║  ┌───────────────┐   ┌────────────────────────┐ ║    │               │ ║
║  │  ║  │ GraphNav      │◄──│   Location Manager     │ ║    │               │ ║
║  │  ║  │ (Waypoints)   │   │   (Save/Load)          │ ║    │               │ ║
║  │  ║  └───────────────┘   └────────────────────────┘ ║    │               │ ║
║  │  ╚═════════════════════════════════════════════════╝    │               │ ║
║  │                                                         │               │ ║
║  │  ╔═════════════════════════════════════════════════╗    │               │ ║
║  │  ║          SAFETY LAYER [COMPLETE]                ║    │               │ ║
║  │  ║  ┌─────────────────────────────────────────────┐║    │               │ ║
║  │  ║  │         E-Stop Service (Always Running)     │╠════╪═══════════════╡ ║
║  │  ║  │              Safety Override                │║    │  OVERRIDE     │ ║
║  │  ║  └─────────────────────────────────────────────┘║    │               │ ║
║  │  ╚═════════════════════════════════════════════════╝    │               │ ║
║  └─────────────────────────────────────────────────────────┘               │ ║
║                                                                            │ ║
╠════════════════════════════════════════════════════════════════════════════╣ ║
║                                                                              ║
║                    SPOT CAPABILITIES (What Users See)                        ║
║                                                                              ║
║  ┌────────────────────────────────────────────────────────────────────────┐  ║
║  │                     ✅ COMPLETE (Green)                                │  ║
║  │                                                                        │  ║
║  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐          │  ║
║  │  │    MOVEMENT     │ │     POSTURE     │ │   NAVIGATION    │          │  ║
║  │  │─────────────────│ │─────────────────│ │─────────────────│          │  ║
║  │  │ • Walk Fwd/Back │ │ • Stand / Sit   │ │ • Go To Location│          │  ║
║  │  │ • Strafe L/R    │ │ • Crouch        │ │ • Save Location │          │  ║
║  │  │ • Turn L/R      │ │ • Stand Tall    │ │ • List Locations│          │  ║
║  │  │ • Turn Around   │ │ • Self-Right    │ │                 │          │  ║
║  │  └─────────────────┘ └─────────────────┘ └─────────────────┘          │  ║
║  │                                                                        │  ║
║  │  ┌─────────────────┐ ┌─────────────────┐                              │  ║
║  │  │     STATUS      │ │     SAFETY      │                              │  ║
║  │  │─────────────────│ │─────────────────│                              │  ║
║  │  │ • Battery Level │ │ • Stop / Halt   │                              │  ║
║  │  │ • Robot Status  │ │ • Freeze        │                              │  ║
║  │  │ • Power Off     │ │ • E-Stop        │                              │  ║
║  │  └─────────────────┘ └─────────────────┘                              │  ║
║  └────────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  ┌────────────────────────────────────────────────────────────────────────┐  ║
║  │                     ⬜ NOT STARTED (Gray)                              │  ║
║  │                                                                        │  ║
║  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐          │  ║
║  │  │ Person Following│ │ PTZ Camera Ctrl │ │ LLM Natural Lang│          │  ║
║  │  │ (Needs Vision)  │ │ (Needs Payload) │ │ (Ollama Ready)  │          │  ║
║  │  └─────────────────┘ └─────────────────┘ └─────────────────┘          │  ║
║  └────────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## Data Flow Summary

```
User Speaks → Microphone → VAD → Noise Reduction → Whisper ASR → Intent Parser → Spot Dispatcher → Robot Action
                                                                        │
                                                                        ├── Movement Commands (SDK)
                                                                        ├── GraphNav (Waypoints)
                                                                        └── Status Queries
```

---

## Component Status Table

| Component | Status | File(s) |
|-----------|--------|---------|
| Voice Activity Detection | ✅ Complete | `client_mic.py` |
| Noise Reduction | ✅ Complete | `client_mic.py` |
| ASR Server (Whisper) | ✅ Complete | `server.py` |
| Intent Parser (Regex) | ✅ Complete | `intent.py` |
| Spot Dispatcher | ✅ Complete | `spot_dispatch.py` |
| E-Stop Service | ✅ Complete | `scripts/estop_run.py` |
| GraphNav Integration | ✅ Complete | `spot_dispatch.py`, `location_manager.py` |
| Movement Commands | ✅ Complete | 8 commands |
| Posture Commands | ✅ Complete | 5 commands |
| Navigation Commands | ✅ Complete | 3 commands |
| Status Commands | ✅ Complete | 3 commands |
| Safety Commands | ✅ Complete | 3 commands |
| Person Following | ⬜ Not Started | - |
| PTZ Camera | ⬜ Not Started | Requires payload |
| LLM Intent Parser | ⬜ Not Started | `intent_llm.py` ready |

---

## Quick Start Flow

```
1. Start E-Stop       →  python scripts/estop_run.py     (Terminal 1 - Keep open!)
2. Start Voice Ctrl   →  python scripts/run_voice_control.py  (Terminal 2)
3. Speak Commands     →  "Stand up" → "Walk forward 2 meters" → "Go to kitchen"
```
