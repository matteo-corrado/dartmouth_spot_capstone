# System Diagram

Every input, through the models and features, to what Spot does, on the poster-demo branch (`demo-poster-2026-10-29`). Solid arrows are the live path; dotted arrows are optional, off by default, or fallbacks. Stage detail with file references is in the [pipeline explainer](pipeline-explainer.md); current state is in [status.md](../project/status.md).

PNG renders: [wide](system-diagram.png) (slides/poster) and [tall](system-diagram-tall.png) (screen reading). Re-render after editing the mermaid source below.

```mermaid
flowchart LR
    %% ---------------- INPUTS ----------------
    subgraph IN["Inputs"]
        MIC["reSpeaker XVF3800 mic<br/>ch1 ASR beam, 16 kHz"]
        XVFCTL["XVF3800 USB control<br/>DoA 20 Hz, per-beam energy"]
        CAMS["Spot body cameras<br/>5 fisheye: front L/R, left, right, back"]
        DEPTH["Front depth<br/>frontleft_depth_in_visual_frame"]
        GRID["Spot LocalGrid<br/>native obstacle map"]
        WOBJ["Spot world objects<br/>AprilTags, dock"]
        RSTATE["Robot state<br/>stance, battery, GraphNav pose"]
        OPS["Operator<br/>spotctl :8770, web panel, E-Stop"]
    end

    %% ---------------- AUDIO FRONT END ----------------
    subgraph AFE["Audio front end (client_mic.py)"]
        VAD["Silero VAD per frame<br/>+ noise floor + endpointer"]
        GATE{"Loudness gate<br/>loudest 150 ms at least 8 dB"}
        SHORT["Short path<br/>up to 1.5 s speech:<br/>commands-only"]
    end

    subgraph KW["Keyword spotters (sherpa-onnx)"]
        SKWS["Safety KWS<br/>stop / freeze<br/>3 streams, always on"]
        WKWS["Wake KWS<br/>HEY SPOT, EXCUSE ME SPOT"]
    end

    subgraph LOCK["Speaker lock"]
        DIR["Direction lock<br/>DoA +/-35 deg, 20 s"]
        VP["Voiceprint<br/>WeSpeaker, off by default"]
    end

    ASR["Parakeet ASR<br/>GPU, gRPC :50055"]
    RULE{"Safety transcript rule<br/>(intent.py)"}

    subgraph BRAIN["Brain (llama-server :11435)"]
        LLM["Gemma 4 E4B Q8<br/>GBNF actions + response<br/>knowledge packs, personas"]
        VLM["Gemma 4 mmproj VLM<br/>first-person scene description"]
    end

    subgraph VISION["Vision models (CPU)"]
        YW["YOLO-World v2<br/>open-vocab object finder"]
        YN["YOLOv8n<br/>person detector"]
    end

    DISP["Dispatcher (spot_dispatch.py)<br/>stop-first, lock-free safety,<br/>emote clearance 1.0 m"]

    %% ---------------- OUTPUTS ----------------
    subgraph OUT["Outputs"]
        MOTION["Locomotion<br/>stand, sit, walk 0.5 m/s cap,<br/>go-to, follow, dances + gestures"]
        STOPOUT["Stop / freeze / stow"]
        ARM["Arm<br/>folded pose, sh0 gaze at speaker,<br/>gripper mouth"]
        TTS["Speech<br/>ElevenLabs Flash, Kokoro fallback<br/>via Spot speaker"]
        RING["LED ring + chimes<br/>idle / listen sweep / think / speak / muted"]
        DET["Detections + reports<br/>scene description, obstacle distance,<br/>object found / lost"]
        WEB["Web panel MJPEG stream"]
        LOGS["Logs<br/>utterances, conversations, latency"]
    end

    %% ---------------- AUDIO FLOW ----------------
    MIC --> VAD
    MIC -->|every frame, before gating| SKWS
    SKWS -->|hit| STOPOUT
    VAD --> GATE
    GATE -->|pass| WKWS
    GATE -->|short + quiet| SHORT
    SHORT --> ASR
    WKWS -->|wake: lock speaker| DIR
    XVFCTL --> DIR
    XVFCTL --> RING
    DIR --> ASR
    VP -.->|"spotctl voiceprint on"| DIR
    MIC -.-> VP
    ASR --> RULE
    RULE -->|stop words| STOPOUT
    RULE -->|everything else| LLM
    LLM -->|actions| DISP
    LLM -->|response text| TTS
    TTS --> ARM

    %% ---------------- CAMERA FLOW ----------------
    CAMS -->|"describe"| VLM
    CAMS -.->|"describe X: hint"| YW
    YW -.-> VLM
    VLM --> TTS
    VLM --> DET
    CAMS -->|"go to object (fallback)"| YW
    YW --> DISP
    CAMS -->|"follow me"| YN
    DEPTH --> YN
    YN --> DISP
    CAMS --> WEB

    %% ---------------- NATIVE PERCEPTION ----------------
    GRID -->|"emote clearance, check obstacles"| DISP
    WOBJ -->|"go to fiducial / dock"| DISP
    RSTATE --> DISP

    %% ---------------- OPERATOR ----------------
    OPS -->|stop, freeze, stow, sit| STOPOUT
    OPS -->|mute, listen, hush, say, persona| AFE
    OPS -.->|E-Stop: cuts motor power| MOTION

    %% ---------------- DISPATCH TO OUTPUTS ----------------
    DISP --> MOTION
    DISP --> STOPOUT
    DISP --> ARM
    DISP --> DET
    DET --> TTS
    LLM -.-> RING
    ASR --> LOGS
    DISP --> LOGS

    classDef safety fill:#fde2e2,stroke:#c0392b,color:#000
    classDef model fill:#e3ecfd,stroke:#2c5aa0,color:#000
    classDef out fill:#e4f6e4,stroke:#2e7d32,color:#000
    class SKWS,RULE,STOPOUT safety
    class WKWS,ASR,LLM,VLM,YW,YN,VP model
    class MOTION,ARM,TTS,RING,DET,WEB,LOGS out
```

## Notes

- **Safety runs beside everything else.** The safety KWS gets every mic frame before the gate, wake word, or mute, and goes straight to stop/freeze. The transcript rule catches stop words ASR heard. Spot never says stop/halt/freeze aloud, so its own voice can't trigger them.
- **Native perception, no cameras:** emote clearance and "check obstacles" read Spot's LocalGrid; fiducial and dock go-to read Spot's world objects (AprilTags). Only follow mode uses depth.
- **Go to object** tries saved GraphNav locations first, then fiducial/dock, and only then the YOLO-World camera search.
- **Not used at runtime:** the hand camera, image grasp, ray-cast helpers and the camera-intrinsics helpers. The XVF3800 AEC does nothing because TTS plays through the Spot speaker, not the mic's own output.
- **Follow mode** (YOLOv8n on fisheye) is unreliable. GPU YOLO26 vision lives on `vision/perception-v2`, not this branch.
