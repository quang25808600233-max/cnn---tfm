#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train CNN-LSTM model với XAUUSD chunks
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import json

print("TensorFlow version:", tf.__version__)
print("GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)
print()

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

CHUNK_DIR = '/workspaces/cnn---tfm/XAUUSD_Chunks'
chunks = sorted([f for f in os.listdir(CHUNK_DIR) if f.endswith('.npz')])

print(f"Found {len(chunks)} chunks")
print()

# Load first chunk for training
chunk_path = os.path.join(CHUNK_DIR, chunks[0])
print(f"Loading {chunks[0]}...")

data = np.load(chunk_path)
X = data['X']  # (N, 60, 12) - sequences
y = data['y']  # (N,) - labels 0/1/2

print(f"  Shape: X={X.shape}, y={y.shape}")
print(f"  Labels: {np.bincount(y)}")
print()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape[0]:,} samples")
print(f"Test:  {X_test.shape[0]:,} samples")
print()

# ══════════════════════════════════════════════════════════════════════════════
# BUILD MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_model(seq_len=60, n_features=12, n_classes=3):
    """CNN-LSTM hybrid model"""
    
    inputs = keras.Input(shape=(seq_len, n_features))
    
    # CNN layers - extract local patterns
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # LSTM layers - capture temporal dependencies
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.3)(x)
    
    # Dense layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

model = build_model()
model.summary()
print()

# ══════════════════════════════════════════════════════════════════════════════
# COMPILE & TRAIN
# ══════════════════════════════════════════════════════════════════════════════

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    ),
    keras.callbacks.ModelCheckpoint(
        'model_best.keras',
        monitor='val_accuracy',
        save_best_only=True
    )
]

print("="*80)
print("  TRAINING")
print("="*80)
print()

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=512,
    callbacks=callbacks,
    verbose=1
)

print()
print("="*80)
print("  EVALUATION")
print("="*80)
print()

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print()

# Predictions
y_pred = model.predict(X_test, verbose=0)
y_pred_class = np.argmax(y_pred, axis=1)

# Confusion matrix
from sklearn.metrics import classification_report, confusion_matrix

print("Classification Report:")
print(classification_report(y_test, y_pred_class, 
                          target_names=['SELL', 'HOLD', 'BUY']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_class)
print("         SELL  HOLD   BUY")
for i, label in enumerate(['SELL', 'HOLD', 'BUY']):
    print(f"{label:4s}  {cm[i, 0]:6d} {cm[i, 1]:6d} {cm[i, 2]:6d}")

print()
print("="*80)
print("  ✅ TRAINING COMPLETE")
print("="*80)
print(f"Model saved: model_best.keras")
