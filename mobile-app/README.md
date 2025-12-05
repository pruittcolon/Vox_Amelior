# Nemo Mobile Companion: Edge Intelligence Client

A production-ready **Flutter** application acting as the multimodal sensory organ for the Nemo Platform. It goes beyond a simple "remote control" to function as an **Edge AI Node**, capable of local wake-word detection, visual buffering, and resilient data synchronization.

---

## 📱 Architectural Overview

The app implements a **Offline-First, Sync-Later** architecture to ensure reliability in varying network conditions.

### Core Capabilities

1.  **Multimodal Ingestion:**
    *   **Audio:** Real-time streaming of Opus-encoded audio to the Nemo Server.
    *   **Vision:** Periodic high-res frame capture for "Vision Mode" context awareness.
2.  **Edge Processing:**
    *   **VAD (Voice Activity Detection):** Runs locally on the device to minimize bandwidth; only streams when speech is detected.
    *   **Wake Word:** Efficient, on-device keyword spotting.
3.  **Resilient Connectivity:**
    *   **BLE Bridge:** Connectivity layer for Smart Glasses / Wearables.
    *   **WebSocket Sync:** Maintains a persistent, bi-directional link for real-time transcription tokens.

---

## ⚡ Key Features

### 1. Vision Mode (Visual Context)
*   **Function:** Captures the user's field of view every 20 seconds.
*   **Implementation:** `CameraController` manages a background capture loop. Images are compressed (JPEG 80%) and sent to the `Vision Service` for object detection and scene description.
*   **Privacy:** Visual data is processed in RAM and discarded; never stored permanently unless explicitly requested ("Remember this").

### 2. Chat Mode (RAG-Enabled)
*   A fully native chat interface (not a WebView).
*   **Context Awareness:** Injects recent transcript history into the chat context, allowing users to ask *"What did I just say?"*.

### 3. Smart Glass Integration
*   **Protocol:** Bluetooth Low Energy (BLE).
*   **Data Flow:** Glasses Mic -> BLE -> Phone (App) -> WiFi -> Nemo Server.
*   **Latency:** Optimized buffer sizes to keep glass-to-server latency under 400ms.

---

## 🛠️ Tech Stack

*   **Framework:** Flutter (Dart)
*   **State Management:** GetX (Reactive State Manager)
*   **Audio:** `soundpool`, `record` (Platform Native Recorder)
*   **Network:** `Dio` (HTTP), `Web_Socket_Channel`
*   **Local DB:** `SQFLite` (Caching transcripts offline)

---

## 🔧 Setup & Installation

### Prerequisites
*   Flutter SDK 3.10+
*   Android Studio / Xcode

### Environment Configuration
The app uses a `.env` file for endpoint configuration.

```bash
# Copy example config
cp .env.example .env

# Edit .env
MEMORY_SERVER_BASE=http://192.168.1.X:8000
```

### Build

```bash
flutter pub get
flutter run --release
```

---

**Pruitt Colon**
*Senior Mobile Engineer*