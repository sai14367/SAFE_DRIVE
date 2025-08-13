import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import threading
import time
import pyttsx3
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import logging
import os
import multiprocessing
import signal

# --- Cross-platform sound setup ---
try:
    from playsound import playsound
    CROSS_PLATFORM_SOUND = True
except ImportError:
    CROSS_PLATFORM_SOUND = False
    print("playsound library not found. Sound features may be limited.")
    if os.name == 'nt':
        try:
            import winsound
            WINSOUND_AVAILABLE = True
        except ImportError:
            WINSOUND_AVAILABLE = False
    else:
        WINSOUND_AVAILABLE = False

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bus_safety_system.log'),
        logging.StreamHandler()
    ]
)

# --- Unified Configuration ---
CONFIG = {
    # Detection parameters
    'EAR_THRESH': 0.20,
    'EAR_CONSEC_FRAMES': 10,
    'MAR_THRESH': 25.0,
    'FACE_MISSING_FRAMES': 50,
    'PHONE_DETECTION_CONFIDENCE': 0.8,
    'MISTAKE_DELAY_FOR_BUZZER': 10,

    # Critical alert parameters
    'STOP_VEHICLE_COUNT_THRESHOLD': 10,
    'STOP_VEHICLE_ALERT_COOLDOWN': 1800,
    'MATH_PROBLEM_COUNT_THRESHOLD': 10,
    'MATH_PROBLEM_COOLDOWN': 3600,
    'MIN_MATH_NUMBER': 1,
    'MAX_MATH_NUMBER': 20,
    'MOVE_ASIDE_COOLDOWN': 600,
    'MOVE_ASIDE_REPETITIONS': 5,
    'MOVE_ASIDE_DELAY': 2,

    # Email settings
    'EMAIL_HOST': "smtp.gmail.com",
    'EMAIL_PORT': 465,
    'EMAIL_USER': "vigneshguthi3212@gmail.com",
    'EMAIL_PASSWORD': "",
    'EMAIL_RECEIVER': "vigneshguthi3212@gmail.com",

    # Sound Settings
    'BEEP_FREQUENCY': 2500,
    'BEEP_DURATION': 1000,
    'WAKEUP_BUZZER_COOLDOWN': 15,
    'BUZZER_SOUND_FILE': 'buzzer_sound.wav',
    
    # Alert Throttling
    'VOICE_ALERT_COOLDOWN_SECONDS': 2,
    'EMAIL_ALERT_COOLDOWN_SECONDS': 300,
}

# --- Helper Functions ---
buzzer_process = None

def play_buzzer_process(sound_file):
    """Plays the buzzer sound in a loop until the process is terminated."""
    try:
        while True:
            playsound(sound_file, block=True)
    except Exception as e:
        pass

def start_buzzer():
    global buzzer_process
    if buzzer_process is None or not buzzer_process.is_alive():
        if os.path.exists(CONFIG['BUZZER_SOUND_FILE']):
            buzzer_process = multiprocessing.Process(target=play_buzzer_process, args=(CONFIG['BUZZER_SOUND_FILE'],))
            buzzer_process.daemon = True
            buzzer_process.start()
        else:
            logging.error(f"Buzzer sound file not found: {CONFIG['BUZZER_SOUND_FILE']}")

def stop_buzzer():
    global buzzer_process
    if buzzer_process is not None and buzzer_process.is_alive():
        try:
            buzzer_process.terminate()
            buzzer_process.join(timeout=1)
        except Exception as e:
            logging.error(f"Error terminating buzzer process: {e}")
        buzzer_process = None

def trigger_alarm(text):
    """Prints and speaks an alert."""
    logging.warning(f"!!! ALERT: {text} !!!")
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        logging.error(f"Text-to-speech engine failed: {e}")

def send_email(subject, body):
    """Sends an email alert."""
    msg = MIMEMultipart()
    msg['From'] = CONFIG['EMAIL_USER']
    msg['To'] = CONFIG['EMAIL_RECEIVER']
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(CONFIG['EMAIL_HOST'], CONFIG['EMAIL_PORT'], context=context) as server:
            server.login(CONFIG['EMAIL_USER'], CONFIG['EMAIL_PASSWORD'])
            server.sendmail(CONFIG['EMAIL_USER'], CONFIG['EMAIL_RECEIVER'], msg.as_string())
        logging.info(f"Email alert sent successfully: {subject}")
    except Exception as e:
        logging.error(f"Email failed to send: {e}")

def eye_aspect_ratio(eye_landmarks, image_shape):
    coords = np.array([(lm.x * image_shape[1], lm.y * image_shape[0]) for lm in eye_landmarks], dtype=np.float32)
    vertical_dist1 = dist.euclidean(coords[1], coords[5])
    vertical_dist2 = dist.euclidean(coords[2], coords[4])
    horizontal_dist = dist.euclidean(coords[0], coords[3])
    if horizontal_dist == 0:
        return 0
    return (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)

def mouth_aspect_ratio(mouth_landmarks, image_shape):
    coords = np.array([(lm.x * image_shape[1], lm.y * image_shape[0]) for lm in mouth_landmarks], dtype=np.float32)
    return dist.euclidean(coords[0], coords[1])

# --- Initialization ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

mp_hands = mp.solutions.hands

LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
MOUTH_INDICES = [13, 14]

# --- Main Detection Loop ---
def main():
    eye_frame_counter = 0
    face_missing_counter = 0
    
    drowsiness_count = 0
    yawn_count = 0
    phone_use_count = 0

    mistake_start_time = {
        'drowsiness': None, 'yawn': None, 'camera_blocked': None, 'phone_call': None
    }
    
    last_alert_time = {
        'drowsiness_voice': 0, 'drowsiness_email': 0,
        'yawn_voice': 0, 'yawn_email': 0,
        'camera_blocked_voice': 0, 'camera_blocked_email': 0,
        'phone_call_voice': 0, 'phone_call_email': 0,
        'stop_vehicle_alert': 0,
        'math_problem': 0,
        'wakeup_buzzer': 0,
        'move_aside_instruction': 0
    }
    
    math_problem_state = {'active': False, 'num1': None, 'num2': None, 'answered': False}
    move_aside_state = {'given': False, 'count': 0, 'last_prompt_time': 0}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.critical("Cannot open camera. Exiting.")
        return

    try:
        with mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
            mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                h, w, _ = frame.shape
                current_time = time.time()

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                face_results = face_mesh.process(rgb_frame)
                hand_results = hands.process(rgb_frame)

                status_color = (0, 255, 0)
                status_text = "STATUS: OK"
                
                is_drowsy = False
                is_yawning = False
                is_phone_detected = False
                is_face_missing = False

                if face_results.multi_face_landmarks:
                    face_missing_counter = 0
                    face_landmarks = face_results.multi_face_landmarks[0]
                    
                    # Draw a bounding box around the detected face
                    face_points = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark], np.int32)
                    x, y, box_w, box_h = cv2.boundingRect(face_points)
                    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (255, 255, 255), 2)
                    
                    ear = (eye_aspect_ratio([face_landmarks.landmark[i] for i in LEFT_EYE_INDICES], (h, w)) +
                           eye_aspect_ratio([face_landmarks.landmark[i] for i in RIGHT_EYE_INDICES], (h, w))) / 2.0
                    mar = mouth_aspect_ratio([face_landmarks.landmark[i] for i in MOUTH_INDICES], (h, w))

                    if hand_results.multi_hand_landmarks:
                        right_ear_landmark = face_landmarks.landmark[132]
                        left_ear_landmark = face_landmarks.landmark[361]
                        phone_roi_right = (int(right_ear_landmark.x * w) - 50, int(right_ear_landmark.y * h) - 50, 100, 100)
                        phone_roi_left = (int(left_ear_landmark.x * w) - 50, int(left_ear_landmark.y * h) - 50, 100, 100)

                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            for landmark_id in [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.THUMB_TIP]:
                                landmark = hand_landmarks.landmark[landmark_id]
                                hand_x, hand_y = int(landmark.x * w), int(landmark.y * h)

                                if (phone_roi_right[0] <= hand_x <= phone_roi_right[0] + phone_roi_right[2] and
                                    phone_roi_right[1] <= hand_y <= phone_roi_right[1] + phone_roi_right[3]) or \
                                   (phone_roi_left[0] <= hand_x <= phone_roi_left[0] + phone_roi_left[2] and
                                    phone_roi_left[1] <= hand_y <= phone_roi_left[1] + phone_roi_left[3]):
                                    is_phone_detected = True
                                    break
                            if is_phone_detected:
                                break

                    if ear <= CONFIG['EAR_THRESH']:
                        eye_frame_counter += 1
                        if eye_frame_counter >= CONFIG['EAR_CONSEC_FRAMES']:
                            is_drowsy = True
                    else:
                        eye_frame_counter = 0

                    if mar > CONFIG['MAR_THRESH']:
                        is_yawning = True

                    cv2.putText(frame, f"EAR: {ear:.2f}", (w-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(frame, f"MAR: {mar:.2f}", (w-150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                else:
                    eye_frame_counter = 0
                    face_missing_counter += 1
                    is_face_missing = True

                # --- Handle Math Problem State ---
                if math_problem_state['active']:
                    status_color = (0, 255, 255)
                    status_text = "STATUS: MATH PROBLEM"
                    problem_text = f"Solve: {math_problem_state['num1']} + {math_problem_state['num2']}"
                    cv2.putText(frame, problem_text, (w // 2 - 200, h // 2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    if buzzer_process is None or not buzzer_process.is_alive():
                        start_buzzer()

                    sum_ans = math_problem_state['num1'] + math_problem_state['num2']
                    sum_text = str(sum_ans)
                    text_size = cv2.getTextSize(sum_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    sum_x = w // 2 - text_size[0] // 2
                    sum_y = h // 2 + 50
                    
                    cv2.putText(frame, sum_text, (sum_x, sum_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    if hand_results.multi_hand_landmarks:
                        hand_landmarks = hand_results.multi_hand_landmarks[0]
                        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        tip_x, tip_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                        
                        cv2.line(frame, (tip_x, tip_y), (sum_x + text_size[0] // 2, sum_y - text_size[1] // 2), (0, 255, 0), 2)
                        
                        if abs(tip_x - (sum_x + text_size[0] // 2)) < 50 and abs(tip_y - (sum_y - text_size[1] // 2)) < 50:
                             logging.info("Driver has indicated the correct answer.")
                             stop_buzzer()
                             math_problem_state['active'] = False
                             math_problem_state['answered'] = True
                             drowsiness_count = 0
                             last_alert_time['math_problem'] = current_time

                # Drowsiness Alerts
                if not math_problem_state['active'] and not move_aside_state['given']:
                    if is_drowsy:
                        if mistake_start_time['drowsiness'] is None:
                            mistake_start_time['drowsiness'] = current_time
                        
                        duration = current_time - mistake_start_time['drowsiness']
                        cv2.putText(frame, "⚠️ DROWSINESS DETECTED! ⚠️", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        status_color = (0, 0, 255)
                        status_text = "STATUS: DROWSINESS ALERT"
                        
                        if current_time - last_alert_time['drowsiness_voice'] > CONFIG['VOICE_ALERT_COOLDOWN_SECONDS']:
                            threading.Thread(target=trigger_alarm, args=("Wake up! Sleeping detected!",)).start()
                            threading.Thread(target=send_email, args=("Sleeping Alert!", "Driver has been detected sleeping.")).start()
                            last_alert_time['drowsiness_voice'] = current_time
                            drowsiness_count += 1
                        
                        if duration > CONFIG['MISTAKE_DELAY_FOR_BUZZER'] and current_time - last_alert_time['wakeup_buzzer'] > CONFIG['WAKEUP_BUZZER_COOLDOWN']:
                             threading.Thread(target=trigger_alarm, args=("Wake up!",)).start()
                             last_alert_time['wakeup_buzzer'] = current_time

                    else:
                        mistake_start_time['drowsiness'] = None
                        last_alert_time['wakeup_buzzer'] = 0
                        
                    if drowsiness_count >= CONFIG['MATH_PROBLEM_COUNT_THRESHOLD'] and current_time - last_alert_time['math_problem'] > CONFIG['MATH_PROBLEM_COOLDOWN']:
                        move_aside_state['given'] = True
                        move_aside_state['count'] = 0
                        
                # State for "Move Vehicle Aside" instruction loop
                if move_aside_state['given'] and not math_problem_state['active']:
                    instruction_text = " MOVE VEHICLE ASIDE "
                    status_text = "STATUS: MOVE ASIDE"
                    status_color = (0, 255, 255)
                    cv2.putText(frame, instruction_text, (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                    
                    if current_time - move_aside_state['last_prompt_time'] > CONFIG['MOVE_ASIDE_DELAY']:
                        if move_aside_state['count'] < CONFIG['MOVE_ASIDE_REPETITIONS']:
                            threading.Thread(target=trigger_alarm, args=(instruction_text,)).start()
                            move_aside_state['last_prompt_time'] = current_time
                            move_aside_state['count'] += 1
                        else:
                            threading.Thread(target=send_email, args=("Safety Instruction: Pull Over!", "Driver has been instructed to pull over to solve a problem.")).start()
                            math_problem_state['active'] = True
                            math_problem_state['num1'] = np.random.randint(CONFIG['MIN_MATH_NUMBER'], CONFIG['MAX_MATH_NUMBER'] + 1)
                            math_problem_state['num2'] = np.random.randint(CONFIG['MIN_MATH_NUMBER'], CONFIG['MAX_MATH_NUMBER'] + 1)
                            math_problem_state['answered'] = False
                            last_alert_time['math_problem'] = current_time
                            last_alert_time['wakeup_buzzer'] = 0
                            move_aside_state['given'] = False
                            move_aside_state['last_prompt_time'] = 0
                
                # Yawn Alerts
                if not math_problem_state['active'] and not move_aside_state['given'] and is_yawning:
                    if mistake_start_time['yawn'] is None:
                        mistake_start_time['yawn'] = current_time
                    
                    cv2.putText(frame, "🥱 YAWN DETECTED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    if status_color == (0, 255, 0):
                        status_color = (0, 255, 255)
                        status_text = "STATUS: YAWN DETECTED"
                    
                    if current_time - last_alert_time['yawn_voice'] > CONFIG['VOICE_ALERT_COOLDOWN_SECONDS']:
                        threading.Thread(target=trigger_alarm, args=("Yawn detected",)).start()
                        threading.Thread(target=send_email, args=("Yawn Alert", "Driver has been detected yawning.")).start()
                        last_alert_time['yawn_voice'] = current_time
                        yawn_count += 1
                else:
                    if not math_problem_state['active'] and not move_aside_state['given']:
                        mistake_start_time['yawn'] = None
                
                # Phone Detection Alerts
                if not math_problem_state['active'] and not move_aside_state['given'] and is_phone_detected:
                    if mistake_start_time['phone_call'] is None:
                        mistake_start_time['phone_call'] = current_time

                    y_pos = 90
                    cv2.putText(frame, "📱 PHONE DETECTED! 📱", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    status_color = (0, 0, 255)
                    status_text = "STATUS: PHONE DETECTED"

                    if current_time - last_alert_time['phone_call_voice'] > CONFIG['VOICE_ALERT_COOLDOWN_SECONDS']:
                        threading.Thread(target=trigger_alarm, args=("Phone detected! Put down the phone!",)).start()
                        threading.Thread(target=send_email, args=("Phone Call Alert!", "Driver has been detected using a phone.")).start()
                        last_alert_time['phone_call_voice'] = current_time
                        phone_use_count += 1
                else:
                    if not math_problem_state['active'] and not move_aside_state['given']:
                        mistake_start_time['phone_call'] = None

                # Camera Blocked / Face Missing Alerts
                if not math_problem_state['active'] and not move_aside_state['given'] and is_face_missing:
                    if mistake_start_time['camera_blocked'] is None:
                        mistake_start_time['camera_blocked'] = current_time
                    
                    cv2.putText(frame, "⚠️ FACE NOT DETECTED ⚠️", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    if face_missing_counter > CONFIG['FACE_MISSING_FRAMES']:
                        cv2.putText(frame, "CAMERA BLOCKED?", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    status_color = (0, 0, 255)
                    status_text = "STATUS: CAMERA BLOCKED/FACE MISSING"

                    if current_time - last_alert_time['camera_blocked_voice'] > CONFIG['VOICE_ALERT_COOLDOWN_SECONDS']:
                        threading.Thread(target=trigger_alarm, args=("Camera Blocked or Face Missing!",)).start()
                        threading.Thread(target=send_email, args=("Camera Alert!", "Driver face is not visible or camera is blocked.")).start()
                        last_alert_time['camera_blocked_voice'] = current_time
                else:
                    if not math_problem_state['active'] and not move_aside_state['given']:
                        mistake_start_time['camera_blocked'] = None

                # Critical "Stop Vehicle" Alert based on cumulative counts
                if not math_problem_state['active'] and not move_aside_state['given']:
                    if (drowsiness_count >= CONFIG['STOP_VEHICLE_COUNT_THRESHOLD'] or
                        yawn_count >= CONFIG['STOP_VEHICLE_COUNT_THRESHOLD'] or
                        phone_use_count >= CONFIG['STOP_VEHICLE_COUNT_THRESHOLD']):
                        
                        if current_time - last_alert_time['stop_vehicle_alert'] > CONFIG['STOP_VEHICLE_ALERT_COOLDOWN']:
                            threading.Thread(target=trigger_alarm, args=("Please stop the vehicle aside!",)).start()
                            threading.Thread(target=send_email, args=("CRITICAL SAFETY ALERT!", "Driver has repeatedly shown signs of distraction/drowsiness. Advise to stop vehicle.")).start()
                            last_alert_time['stop_vehicle_alert'] = current_time

                # Display Counts on Screen
                cv2.putText(frame, f"Drowsy Events: {drowsiness_count}", (w - 280, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Yawn Events: {yawn_count}", (w - 280, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Phone Events: {phone_use_count}", (w - 280, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # Display overall status
                cv2.putText(frame, status_text, (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
        
                cv2.imshow("Bus Safety System", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_buzzer()
                    break
    finally:
        logging.info("Shutting down system.")
        stop_buzzer()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
