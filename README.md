# SAFE_DRIVE
Here’s a polished, production-ready **README.md** you can drop into your repo for the bus/driver safety system you shared.

---

# Driver Drowsiness, Yawn & Distraction Alert (OpenCV + MediaPipe)

Real-time driver monitoring that detects **drowsiness (EAR)**, **yawning (MAR)**, **phone use near ears (hand landmarks)**, **face missing/camera blocked**, and escalates with **voice prompts, email alerts, looping buzzer**, and a **“pull over + quick math” recovery flow**.



> Works on a standard webcam. Built with **Python**, **OpenCV**, **MediaPipe**, and lightweight heuristics (no GPU required).

---

## ✅ Features

* **Drowsiness detection** with Eye Aspect Ratio (EAR) + consecutive frame logic
* **Yawn detection** using a simple Mouth Aspect Ratio (MAR) threshold
* **Phone-near-ear detection** with MediaPipe Hands + face ear ROIs
* **Face missing / camera blocked** detection
* **Escalation pipeline**

  * Voice prompts (TTS)
  * Optional **email alerts** (Gmail SMTP)
  * Optional **looping buzzer** (separate process)
  * “**Move vehicle aside**” repeated instruction
  * **Quick math problem** interaction to verify alertness
* **On-screen telemetry**: status banner + EAR/MAR + event counters
* Robust throttling/cooldowns to avoid alert spam
* Cross-platform sound support (`playsound`, with Windows `winsound` fallback)

---

## 🧰 Tech Stack

* **Python 3.10+**
* **OpenCV** (video I/O & drawing)
* **MediaPipe** (FaceMesh & Hands)
* **NumPy, SciPy** (geometry & distances)
* **pyttsx3** (offline TTS)
* **playsound** (looping buzzer, optional)
* **smtplib / ssl** (built-in) for email alerts
* **multiprocessing**, **threading** for non-blocking UX

---

## 📁 Project Structure

```
.
├─ main.py                    # Entry point
├─ requirements.txt           # (optional) see deps below
├─ bus_safety_system.log      # runtime log (created at run)
├─ buzzer_sound.wav           # buzzer sound (you provide)
└─ README.md
```

---

## 🔧 Installation

1. **Clone & enter**

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

2. **Create a virtual env (recommended)**

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install opencv-python mediapipe numpy scipy pyttsx3 playsound
```

> `playsound` is optional but recommended for the buzzer loop.
> `smtplib` and `ssl` are part of Python’s standard library.

---

## 🔐 Email (Gmail) Setup (Optional)

If you want email alerts:

* Set `CONFIG['EMAIL_USER']` and `CONFIG['EMAIL_RECEIVER']` to your addresses.
* For Gmail, create an **App Password** (under Google Account → Security → App passwords) and put it in `CONFIG['EMAIL_PASSWORD']`.
* Port `465` with `SMTP_SSL` is already configured.

> If you skip email, leave the password blank—the code will just log the failure and keep running.

---

## ⚙️ Configuration (in `main.py`)

All tunables live in the `CONFIG` dict:
```

CONFIG = {
    # Detection thresholds
    'EAR_THRESH': 0.20,               # eye-closed threshold (lower = stricter)
    'EAR_CONSEC_FRAMES': 10,          # frames below EAR -> drowsy
    'MAR_THRESH': 25.0,               # yawn open distance (pixel scale via landmarks)
    'FACE_MISSING_FRAMES': 50,        # frames without face -> camera blocked?
    'PHONE_DETECTION_CONFIDENCE': 0.8,# reserved knob (ROI-based phone heuristic)
    'MISTAKE_DELAY_FOR_BUZZER': 10,   # seconds before wakeup buzzer

    # Escalation/critical
    'STOP_VEHICLE_COUNT_THRESHOLD': 10,
    'STOP_VEHICLE_ALERT_COOLDOWN': 1800,  # seconds
    'MATH_PROBLEM_COUNT_THRESHOLD': 10,
    'MATH_PROBLEM_COOLDOWN': 3600,
    'MIN_MATH_NUMBER': 1,
    'MAX_MATH_NUMBER': 20,
    'MOVE_ASIDE_COOLDOWN': 600,
    'MOVE_ASIDE_REPETITIONS': 5,
    'MOVE_ASIDE_DELAY': 2,

    # Email (optional)
    'EMAIL_HOST': "smtp.gmail.com",
    'EMAIL_PORT': 465,
    'EMAIL_USER': "you@gmail.com",
    'EMAIL_PASSWORD': "",              # <- app password
    'EMAIL_RECEIVER': "you@gmail.com",

    # Sound
    'BEEP_FREQUENCY': 2500,
    'BEEP_DURATION': 1000,
    'WAKEUP_BUZZER_COOLDOWN': 15,
    'BUZZER_SOUND_FILE': 'buzzer_sound.wav',

    # Alert throttles
    'VOICE_ALERT_COOLDOWN_SECONDS': 2,
    'EMAIL_ALERT_COOLDOWN_SECONDS': 300,
}
```

> You can start with defaults and tweak **EAR**, **MAR**, and **consecutive frames** based on your camera and lighting. See **Tuning** below.


## ▶️ Run

python main.py


* Press **`q`** to quit.
* A window titled **“Bus Safety System”** will show:

  * **Status banner** (OK / Drowsiness / Yawn / Phone / Blocked / Math / Move Aside)
  * **EAR** & **MAR** values
  * **Event counters** (Drowsy / Yawn / Phone)


## 🧠 How It Works

### 1) Face + Eye + Mouth (MediaPipe FaceMesh)

* Extracts eyes (fixed landmark indices) and computes **EAR**:

  $$
  EAR = \frac{\|p_2-p_6\| + \|p_3-p_5\|}{2 \cdot \|p_1-p_4\|}
  $$

* Detects drowsiness if `EAR <= EAR_THRESH` for `EAR_CONSEC_FRAMES`.

* Computes a simple **MAR** (mouth opening distance between upper/lower lip landmarks).
  If `MAR > MAR_THRESH`, we mark **yawning**.

> MAR here is a **simple pixel distance**; you can replace it with a normalized ratio if you want invariance to scale.

### 2) Phone-near-ear (MediaPipe Hands + Face ROIs)

* Tracks hands and checks **thumb/index tips** entering **ear-centered ROIs** (left/right).
* If hand enters either ROI → **phone detected** (a good heuristic for on-call posture).

### 3) Escalation Logic

* Voice prompts (pyttsx3) + throttling
* Email notifications (optional)
* If repeated issues:

  * “**Move vehicle aside**” loop with repeated voice reminders
  * Then starts a **quick math problem** (e.g., `7 + 13`), shows the **answer on screen** and asks the driver to point near it with a finger (hand tracking) to confirm engagement.
* Optional **buzzer loop** (separate process) during the math phase

---

## 🎛️ Tuning Tips

* **EAR** (`0.18–0.25` typical). If you get false positives, **lower** it slightly; if you miss closures, **raise** it.
* **EAR\_CONSEC\_FRAMES**: With 30 FPS, `10–20` frames ≈ `0.33–0.66s`. Increase to require longer eye closures.
* **MAR\_THRESH**: Depends on your camera scale. Start at `25.0`, then watch the on-screen MAR value as you yawn and set a **threshold a bit below typical yawn peaks**.
* Good lighting and a camera at **eye level** help a lot.

---

## 🧪 Optional: Improving Yawn Accuracy with a Dataset

If you want higher accuracy than a simple MAR threshold:

1. Collect/prepare a dataset (e.g., **YawDD** on Kaggle) split into:

   ```
   dataset/
     yawn/
     no_yawn/
   ```
2. Crop the **mouth ROI** per frame (FaceMesh), resize to 64×64, normalize.
3. Train a small CNN classifier (binary: yawn vs no\_yawn).
4. Replace the MAR check with `model.predict(mouth_roi)` and apply a confidence threshold.

*(This keeps runtime light while improving robustness.)*

---

## 🐛 Troubleshooting

* **No window / black screen**: ensure your webcam is free and `cv2.VideoCapture(0)` opens successfully.
* **No sound**:

  * Ensure `buzzer_sound.wav` exists and `playsound` is installed.
  * pyttsx3 requires an available speech engine (Windows SAPI is built-in).
* **Email fails**:

  * Use a **Gmail App Password** (not your normal Gmail password).
  * Check firewall/antivirus rules for Python.
* **High CPU**:

  * Lower camera resolution (e.g., `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)` and `CAP_PROP_FRAME_HEIGHT, 480)`).
  * Increase throttles/cooldowns, or reduce detection frequency.

---

## 📈 Roadmap

* Normalize **MAR** using inner mouth height / mouth width for scale invariance
* Add **head pose** estimation (nodding off)
* Integrate a **tiny mouth-ROI CNN** for yawning
* **Twilio SMS** integration for critical alerts
* Config file (`yaml/json`) instead of in-code dict

---

## 🤝 Contributing

Issues and PRs are welcome!
Got a new heuristic or a tiny CNN that works better? Share it 🙌

---

## 📄 License

MIT — do whatever, just keep the notice. Drive safely 🛡️

---

## ✍️ Credits

* **Google MediaPipe** (FaceMesh & Hands)
* **OpenCV** for video & drawing
* Inspiration from EAR/MAR literature for lightweight driver monitoring

---
