# How to Wake This Device Remotely

This device ("Nemo Server") is configured to wake up when it receives a "Magic Packet" over WiFi.

## 1. Setup on Nemo Server (This Device)
(This step should be done by the `setup_wowlan.sh` script on the server itself)
- Connection Name: `WuTang_Lan`
- Interface: `wlo1`
- MAC Address: `80:32:53:22:c1:35`

## 2. Setup on Main Computer (Remote Controller)

To wake this device, you need to send a Magic Packet to the MAC address `80:32:53:22:c1:35`.

### Option A: Use the included Python Script
I have placed the script `wake_device.py` in this shared folder. You can run it from your Main Computer if you have this folder mounted.

```bash
python3 /path/to/mounted/share/wake_device.py
```

### Option B: "Magic Word" Voice Trigger (Linux/Mac/Windows)
If you want to use a "Magic Word" to turn it on (e.g., "Computer, Wake Up"), you can use a simple speech recognition loop.

**Example Python Script (requires `SpeechRecognition` and `pyaudio`):**

```python
import speech_recognition as sr
import os

# The exact phrase you want to use
MAGIC_WORD = "wake up nemo"

def listen_for_magic_word():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print(f"Listening for '{MAGIC_WORD}'...")
        while True:
            try:
                audio = r.listen(source, timeout=5, phrase_time_limit=5)
                text = r.recognize_google(audio).lower()
                print(f"Heard: {text}")
                if MAGIC_WORD in text:
                    print("Magic word detected! Sending wake command...")
                    os.system("python3 wake_device.py")
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    listen_for_magic_word()
```

### Option C: Mobile Apps
You can also use any "Wake on LAN" app on your phone. Just enter the MAC address `80:32:53:22:c1:35`.
