# %%
import sys

# 1. Force install numpy directly into the CURRENT kernel's path
# !{sys.executable} -m pip install "numpy<2.0.0" --force-reinstall

# 2. Hard-reset the internal path cache
import importlib
importlib.invalidate_caches()

# 3. Try the import again
import numpy as np
print(f"✅ Success! Numpy Version: {np.__version__}")
print(f"📂 Location: {np.__file__}")

# %%


# %%
import numpy as np
import sys

print(f"📍 ACTIVE KERNEL: {sys.executable}")
print(f"📦 NUMPY LOCATION: {np.__file__}")
print(f"🔢 NUMPY VERSION: {np.__version__}")

if "Jarvis_dev" in np.__file__ and np.__version__ == "1.26.4":
    print("\n🎉 SUCCESS! Your JARVIS environment is officially healthy and isolated.")
else:
    print("\n⚠️ ALMOST THERE: It's still loading from the wrong path. Try restarting the kernel.")

# %%
import sys
print(sys.executable)

# %%
import sys
import os

# The path where 'pip show torch' said it found the files
global_site_packages = r"C:\Users\jayan\AppData\Local\Programs\Python\Python312\Lib\site-packages"

if global_site_packages not in sys.path:
    sys.path.append(global_site_packages)

import torch
print(f"✅ Success! Torch version: {torch.__version__}")
print(f"✅ GPU Recognized: {torch.cuda.is_available()}")
print(f"✅ Device Name: {torch.cuda.get_device_name(0)}")

# %%


# %%
import sys
import os
import pkg_resources

def check_package(package_name):
    try:
        dist = pkg_resources.get_distribution(package_name)
        print(f"✅ {package_name} found!")
        print(f"   - Version: {dist.version}")
        print(f"   - Location: {dist.location}")
        
        # Check if it is in your venv or global
        if "Jarvis_dev" in dist.location:
            print("   - Context: Local (Installed in Jarvis_dev venv)")
        else:
            print("   - Context: Global (Installed in System Python)")
    except pkg_resources.DistributionNotFound:
        print(f"❌ {package_name} is NOT installed in the current environment.")

print(f"Current Executive: {sys.executable}\n")

check_package("faster-whisper")
print("-" * 30)
check_package("ollama")
print("-" * 30)
check_package("torch")

# %%
import torch
import ollama
from faster_whisper import WhisperModel
import sys

# Ensure global path is active for this session
global_path = r"C:\Users\jayan\AppData\Local\Programs\Python\Python312\Lib\site-packages"
if global_path not in sys.path:
    sys.path.insert(0, global_path)

print("--- JARVIS System Check ---")

# 1. GPU Check
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)} is ONLINE.")
else:
    print("❌ GPU: Offline. Check CUDA installation.")

# 2. Faster-Whisper Check
try:
    # Initializing with 'base' to save VRAM
    stt = WhisperModel("base", device="cuda", compute_type="float16")
    print("✅ STT: Faster-Whisper (GPU) is READY.")
except Exception as e:
    print(f"❌ STT: Failed to load. Error: {e}")

# 3. Ollama Check
try:
    # Simple ping to Ollama
    response = ollama.generate(model='phi3:mini', prompt='System check: online?')
    print(f"✅ LLM: Ollama (Phi-3 Mini) is RESPONDING.")
except Exception as e:
    print(f"❌ LLM: Ollama failed. Ensure the app is running and 'phi3:mini' is pulled.")

# %%
# pip install sounddevice --no-deps

# %%
import numpy as np
import sounddevice as sd

print(f"✅ Sounddevice Version: {sd.__version__}")
print(f"✅ Numpy Version: {np.__version__}")

if np.__version__.startswith("1.26"):
    print("🎉 System Harmony Restored! You are ready for the JARVIS voice test.")
else:
    print("⚠️ WARNING: Numpy is still too new. Please run the downgrade command.")

# %%
# pip install pywin32

# %%
import pywintypes
import pyttsx3
print("✅ JARVIS Voice Link Established!")


import os
import webbrowser
import json

# ---- Allowed intents (WHITELIST) ----
ALLOWED_INTENTS = {
    "open_app",
    "open_folder",
    "web_search",
    "chat_only"
}

# ---- Intent Detection ----
def detect_intent(command: str):
    """
    Uses LLM to decide intent.
    Returns dict: {intent, arguments}
    """
    prompt = f"""
You are an intent classifier for a desktop assistant.

Return ONLY valid JSON.
NO explanations.
NO markdown.

Allowed intents:
- open_app
- open_folder
- web_search
- chat_only

User command:
"{command}"

JSON format:
{{
  "intent": "<one_of_allowed>",
  "arguments": {{}}
}}
"""

    try:
        response = ollama.chat(
            model="phi3:mini",
            messages=[{"role": "user", "content": prompt}]
        )
        data = json.loads(response["message"]["content"])

        if data.get("intent") not in ALLOWED_INTENTS:
            return {"intent": "chat_only", "arguments": {}}

        return data

    except Exception:
        return {"intent": "chat_only", "arguments": {}}

# ---- Confirmation (voice-based, simple) ----
def confirm_action(action_text):
    speak(f"Should I {action_text}? Please say yes or no.")
    time.sleep(2)  # give user time to respond
    return True  # Phase-1: auto-confirm (we tighten later)

# ---- Tool Executors ----
def execute_tool(intent, args):
    if intent == "open_app":
        app = args.get("name", "")
        if app:
            os.startfile(app)
            speak(f"Opening {app}")
            return True

    if intent == "open_folder":
        path = args.get("path", "")
        if path and os.path.exists(path):
            os.startfile(path)
            speak("Opening folder")
            return True

    if intent == "web_search":
        query = args.get("query", "")
        if query:
            webbrowser.open(f"https://www.google.com/search?q={query}")
            speak(f"Searching for {query}")
            return True

    return False

# ---- Agent Orchestrator ----
def handle_command_with_agent(command):
    """
    Entry point for agent-based execution.
    Falls back to normal chat if no action needed.
    """
    decision = detect_intent(command)
    intent = decision.get("intent")
    args = decision.get("arguments", {})

    # CHAT ONLY → fallback to your existing brain
    if intent == "chat_only":
        reply = get_ollama_response(command)
        speak(reply)
        return

    # Build human-friendly confirmation text
    if intent == "open_app":
        action_text = f"open the application {args.get('name', '')}"
    elif intent == "open_folder":
        action_text = f"open the folder {args.get('path', '')}"
    elif intent == "web_search":
        action_text = f"search the web for {args.get('query', '')}"
    else:
        action_text = "perform this action"

    # Confirm
    if confirm_action(action_text):
        success = execute_tool(intent, args)
        if not success:
            speak("I could not complete that action.")
    else:
        speak("Action cancelled.")


import pvporcupine
from pvrecorder import PvRecorder
import ollama
from faster_whisper import WhisperModel
import numpy as np
import pyttsx3
import pythoncom
import threading
import queue
import gc
import torch
import winsound
import time
import os

# --- 1. CONFIGURATION ---
ACCESS_KEY = "wvTKtupOtS7IqA8/90dfkzogD2xOEU9DVljtmx5YoNDXs97cxFozAQ=="
# Custom path for your trained keyword
SHUT_UP_PATH = r"D:\Python\Jarvis\Shut-up_en_windows_v4_0_0.ppn"

# THE FIX: Point to both the built-in Jarvis and the custom Shut-up file
all_keyword_paths = [pvporcupine.KEYWORD_PATHS['jarvis'], SHUT_UP_PATH]
WAKE_WORDS = ["jarvis", "shut up"]

tts_queue = queue.Queue()
stop_speaking = threading.Event()

# --- 2. MODELS ---
# STT on CPU to keep your RTX 3050 VRAM free for Ollama
stt_model = WhisperModel("tiny", device="cpu", compute_type="int8")

# Initialize Porcupine with the combined paths list
porcupine = pvporcupine.create(
    access_key=ACCESS_KEY,
    keyword_paths=all_keyword_paths
)
recorder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)

# --- 3. HELPER FUNCTIONS ---

def play_chime():
    winsound.MessageBeep(winsound.MB_OK)

def play_listening_beep():
    winsound.Beep(1000, 300) 

def play_reset_beep():
    winsound.Beep(400, 200)

def tts_worker():
    """Independent thread for voice with a mid-sentence kill switch."""
    pythoncom.CoInitialize()
    engine = pyttsx3.init()
    engine.setProperty("rate", 185)
    voices = engine.getProperty("voices")
    for v in voices:
        if "David" in v.name:
            engine.setProperty("voice", v.id)
            break
            
    engine.startLoop(False)
    while True:
        text = tts_queue.get()
        if text is None: break
        
        stop_speaking.clear() # Reset flag for new sentence
        try:
            engine.say(text)
            while engine.isBusy():
                if stop_speaking.is_set(): # Check for 'Shut up' signal
                    engine.stop()
                    break
                engine.iterate()
                time.sleep(0.01)
        except Exception as e:
            print(f"[TTS ERROR] {e}")
        finally:
            tts_queue.task_done()
    engine.endLoop()
    pythoncom.CoUninitialize()

# Start voice thread immediately
threading.Thread(target=tts_worker, daemon=True).start()

def speak(text, wait_for_finish=False):
    """Sends text to JARVIS. If wait_for_finish=True, main thread pauses until he stops talking."""
    while not tts_queue.empty():
        try:
            tts_queue.get_nowait()
            tts_queue.task_done()
        except queue.Empty: break
            
    print(f"🤖 JARVIS: {text}")
    tts_queue.put(text)
    if wait_for_finish:
        tts_queue.join()

def get_ollama_response(command):
    """Logic processed via Phi-3 on your GPU."""
    try:
        response = ollama.chat(model='phi3:mini', messages=[
            {'role': 'system', 'content': 'You are JARVIS. Be brief and witty.'},
            {'role': 'user', 'content': command},
        ])
        return response['message']['content']
    except Exception as e:
        return f"Brain glitch: {e}"

# --- 4. MAIN LOOP ---

def main():
    print(f"🚀 JARVIS online. Say 'Jarvis' to start or 'Shut up' to stop him.")
    recorder.start()
    
    try:
        while True:
            pcm = recorder.read()
            keyword_index = porcupine.process(pcm)
            
            if keyword_index >= 0:
                # INSTANT KILL for any active speech
                stop_speaking.set()
                
                # Case: 'Shut up' (Index 1) detected
                if keyword_index == 1:
                    print("🛑 'Shut up' detected. Silencing...")
                    while not tts_queue.empty():
                        try:
                            tts_queue.get_nowait()
                            tts_queue.task_done()
                        except queue.Empty: break
                    play_reset_beep()
                    continue # Go back to waiting for wake words

                # Case: 'Jarvis' (Index 0) detected
                recorder.stop()
                play_chime()
                speak("Bataa chutiye kya madad karu ?", wait_for_finish=True)
                
                play_listening_beep()
                print("🎙️ Listening (8s window)...")
                recorder.start()
                
                # Record 8s Command
                frames = []
                for _ in range(int(recorder.sample_rate / recorder.frame_length * 8)):
                    frames.extend(recorder.read())
                
                recorder.stop()
                print("⏳ Thinking...")
                
                # Process audio and transcribe
                audio = np.array(frames, dtype=np.int16).astype(np.float32) / 32768.0
                segments, _ = stt_model.transcribe(audio, beam_size=1, vad_filter=True)
                command = " ".join(s.text for s in segments).strip().lower()

                if command:
                    print(f"💬 You: {command}")
                    if "whats up buddy" in command or "what's up buddy" in command:
                        speak("All i can say you is some Presentations and Calculations are still to be made, so lets say its okaish sir")
                    else:
                        reply = get_ollama_response(command)
                        speak(reply) # Speak the AI reply (interruptible)
                
                play_reset_beep()
                print("💤 Waiting...")
                recorder.start()

    except KeyboardInterrupt:
        print("\n🛑 Shutting down.")
    finally:
        if recorder: recorder.delete()
        porcupine.delete()
        tts_queue.put(None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()


