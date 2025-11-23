import os, glob, numpy as np, cv2, mediapipe as mp

# Path to your dataset
SRC = "datasets/SignLanguageDigitsDataset"

OUT_DIR = "project/data"
os.makedirs(OUT_DIR, exist_ok=True)

mp_hands = mp.solutions.hands

allowed_ext = ["jpg", "jpeg", "png"]

def normalize_hand(pts):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    diag = ((maxx - minx)**2 + (maxy - miny)**2)**0.5 or 1e-6
    wx, wy = pts[0]
    feat = []
    for (x, y) in pts:
        feat += [(x - wx) / diag, (y - wy) / diag]
    return feat

def extract_features(img, hands):
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if not res.multi_hand_landmarks:
        return None
    pts = [(lm.x, lm.y) for lm in res.multi_hand_landmarks[0].landmark]
    return normalize_hand(pts)

X, y = [], []

print("Starting preprocessing...")

with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
    for digit in range(10):
        digit_folder = os.path.join(SRC, str(digit))

        # The images are inside a subfolder named "Input Images - Sign X"
        # So find any folder containing "Input"
        subfolders = [f for f in os.listdir(digit_folder) 
                      if os.path.isdir(os.path.join(digit_folder, f)) 
                      and "Input" in f]

        if not subfolders:
            print(f"No input folder found in {digit_folder}")
            continue
        
        img_folder = os.path.join(digit_folder, subfolders[0])
        print(f"Processing digit {digit} in {img_folder}")

        count = 0
        for ext in allowed_ext:
            for f in glob.glob(os.path.join(img_folder, f"*.{ext}")):
                img = cv2.imread(f)
                feat = extract_features(img, hands)
                if feat is not None:
                    X.append(feat)
                    y.append(digit)
                count += 1

        print(f"   Found {count} image files for digit {digit}")

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print("Final shapes:", X.shape, y.shape)

np.save(os.path.join(OUT_DIR, "X.npy"), X)
np.save(os.path.join(OUT_DIR, "y.npy"), y)

print("✅ Saved features to project/data/")
