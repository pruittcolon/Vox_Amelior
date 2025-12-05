# Vision Mode

Vision Mode is a specialized feature of the Nemo Mobile Companion app that utilizes the device's camera to provide multimodal context to the AI.

## Functionality

- **Periodic Capture:** Automatically captures an image every 20 seconds.
- **Zoom Control:** Sets camera zoom to ~2.1x (optimized for wearable perspective).
- **AI Analysis:** Sends captured images to the Nemo Server for processing.
- **Context Integration:** Used to enhance "System 2" thinking by providing visual ground truth to the conversation context.

## Usage

1. Open the app.
2. Navigate to the **Vision Mode** tab.
3. Grant camera permissions if requested.
4. The app will begin the capture loop automatically.
5. View logs in the `AppLogger` to confirm transmission.

## Technical Details

- **File:** `lib/views/vision_mode_page.dart`
- **Camera Plugin:** `camera` (Flutter Official)
- **Logic:**
    - Initializes camera on page load.
    - Starts a recurring `Timer` (20s).
    - Silently captures a frame.
    - Uploads to `POST /api/vision/analyze` (or equivalent endpoint).
