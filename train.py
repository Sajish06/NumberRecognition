import os, numpy as np, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = "project/data"
MODEL_DIR = "project/models"
os.makedirs(MODEL_DIR, exist_ok=True)

X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))  # labels 0..9

# train/val split
Xtr, Xva, ytr, yva = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

classes = np.unique(ytr)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=ytr)
class_weights = {int(c): float(w) for c, w in zip(classes, class_weights)}

inp = tf.keras.Input(shape=(X.shape[1],))
x = tf.keras.layers.Dense(256, activation='relu')(inp)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
out = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inp, out)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

ckpt = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(MODEL_DIR, "hand_digits_mlp.keras"),
    monitor="val_accuracy", save_best_only=True
)
es = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=10, restore_best_weights=True
)

history = model.fit(
    Xtr, ytr,
    validation_data=(Xva, yva),
    epochs=80,
    batch_size=64,
    class_weight=class_weights,
    callbacks=[ckpt, es],
    verbose=1
)

val_loss, val_acc = model.evaluate(Xva, yva, verbose=0)
print(f"Validation accuracy: {val_acc:.3f}")
