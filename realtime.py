import os, cv2, numpy as np, mediapipe as mp, tensorflow as tf
from collections import deque

TRY_TTS = True
tts_engine = None
if TRY_TTS:
    try:
        import pyttsx3
        tts_engine = pyttsx3.init()
    except Exception:
        tts_engine = None

MODEL_PATH = "project/models/hand_digits_mlp.keras"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def normalize_hand(pts):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    diag = ((maxx - minx)**2 + (maxy - miny)**2) ** 0.5 or 1e-6
    wx, wy = pts[0]
    feat = []
    for (x, y) in pts:
        feat += [(x - wx) / diag, (y - wy) / diag]
    return feat

def extract_features(results):
    # Expecting only ONE hand (dataset is single-hand)
    if not results.multi_hand_landmarks:
        return np.zeros(42, dtype=np.float32)

    # Use only the first detected hand
    hand = results.multi_hand_landmarks[0]
    pts = [(lm.x, lm.y) for lm in hand.landmark]

    # Normalize
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    diag = ((maxx - minx)**2 + (maxy - miny)**2)**0.5 or 1e-6

    wx, wy = pts[0]
    feat = []
    for (x, y) in pts:
        feat.append((x - wx) / diag)
        feat.append((y - wy) / diag)

    return np.array(feat, dtype=np.float32)  # (42,)

def speak(text):
    if tts_engine:
        try:
            tts_engine.say(text); tts_engine.runAndWait()
        except Exception:
            pass

def main():
    # load model
    model = tf.keras.models.load_model(MODEL_PATH)

    cap = cv2.VideoCapture(0)  # if you have multiple cams, try 1, 2, ...
    if not cap.isOpened():
        raise RuntimeError("Webcam not found")

    # optional record to file
    recording = False
    writer = None

    # smoothing
    pred_window = deque(maxlen=12)
    stable_digit = None
    buffer_digits = []
    use_speech = False

    print("Controls: q=quit | s=toggle speech | r=toggle recording | space=append current digit | backspace=delete | enter=clear")
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                        model_complexity=1, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for lms in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

            feat = extract_features(results).reshape(1, -1)
            probs = model.predict(feat, verbose=0)[0]  # 10 probs
            pred = int(np.argmax(probs))  # 0..9
            conf = float(np.max(probs))

            pred_window.append(pred)
            vals, cts = np.unique(pred_window, return_counts=True)
            maj = int(vals[np.argmax(cts)]) if len(vals) > 0 else None

            if maj is not None and maj != stable_digit and cts.max() >= max(5, len(pred_window)*0.6) and conf > 0.6:
                stable_digit = maj
                if use_speech:
                    speak(str(stable_digit))

            # UI overlay
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)
            cv2.putText(frame, f"Pred: {stable_digit if stable_digit is not None else '-'}", (12, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
            cv2.putText(frame, f"Conf: {conf:.2f}", (240, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.putText(frame, "Buffer: " + "".join(str(d) for d in buffer_digits), (12, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
            cv2.putText(frame, f"TTS: {'ON' if use_speech else 'OFF'}  REC: {'ON' if recording else 'OFF'}", (12, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

            cv2.imshow("Live 0-9 Identifier", frame)
            k = cv2.waitKey(1) & 0xFF

            if k == ord('q'):
                break
            elif k == ord('s'):
                use_speech = not use_speech
            elif k == ord('r'):
                recording = not recording
                if recording and writer is None:
                    os.makedirs("project/demos", exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter("project/demos/demo.mp4", fourcc, 24.0, (w, h))
                elif not recording and writer is not None:
                    writer.release(); writer = None
            elif k == 32:  # space
                if stable_digit is not None:
                    buffer_digits.append(stable_digit)
                    if use_speech:
                        speak(str(stable_digit))
            elif k == 8:  # backspace
                if buffer_digits:
                    buffer_digits.pop()
            elif k == 13: # enter
                buffer_digits.clear()

            if recording and writer is not None:
                writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
