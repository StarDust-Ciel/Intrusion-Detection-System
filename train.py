import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, f1_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

model = os.path.expanduser("~/nids-project/models")
data1 = os.path.expanduser("~/nids-project/data")
os.makedirs(model, exist_ok=True)

print("Loading preprocessed data.")
X,y = joblib.load(os.path.join(data1, "processed.pkl"))
scaler = joblib.load(os.path.join(model, "scaler.pkl")) 
label_encoder = joblib.load(os.path.join(model, "label_encoder.pkl")) 
class_names = list(label_encoder.classes_)
n_classes = len(class_names)

feature_names = joblib.load(os.path.join(model, 'feature_names.pkl')) 
X = X[feature_names] 
X_scaled = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=0.2, random_state=42, stratify=y)
print(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
print(f"Classes: {class_names}\n")


def save_confusion_matrix(y_true, y_pred, model_name):
   cm = confusion_matrix(y_true, y_pred)
   cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
   fig, ax = plt.subplots(figsize=(10,8))
   sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names,ax=ax)
   ax.set_xlabel('Predicted')
   ax.set_ylabel('Actual')
   ax.set_title(f"Confusion matrix {model_name}")
   plt.tight_layout()
   out_path = os.path.join(model, f'confusion_{model_name.lower().replace(" ","_")}.png')
   plt.savefig(out_path, dpi=150)
   plt.close()
   print(f"Confusion matrix saved in {out_path}")

print("Deep neural network: ")
model_nn = Sequential([
   Dense(256, activation ='relu', input_shape=(X_train.shape[1],)),
   BatchNormalization(),
   Dropout(0.3),
   Dense(128, activation='relu'),
   Dropout(0.2),
   Dense(64, activation='relu'),
   Dense(n_classes, activation='softmax')
    ])

model_nn.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_nn.fit(X_train, y_train, epochs=30, batch_size=1024, validation_split=0.2, callbacks=[early_stop], verbose=1)
nn_preds = np.argmax(model_nn.predict(X_test, verbose=0), axis=1)
nn_acc = accuracy_score(y_test, nn_preds)
nn_f1 = f1_score(y_test, nn_preds, average='weighted')
print(f"\n Neural Network Accuracy: {nn_acc:.4f} | Weighted F1:{nn_f1:.4f}")
print(classification_report(y_test,nn_preds, target_names=class_names))
save_confusion_matrix(y_test, nn_preds,"Neural Network")
model_nn.save(os.path.join(model,'nids_model_nn.h5'))
print(f"Model saved in {model}/nids_model_nn.h5\n") 

print("Random Forest: ")
rf = RandomForestClassifier(
    n_estimators=100,   
    max_depth=20,       
    n_jobs=2,           
    random_state=42,
    verbose=1
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
rf_f1 = f1_score(y_test, rf_preds, average='weighted')

print(f"\n Random Forest Accuracy: {rf_acc:.4f} | Weighted F1:{rf_f1:.4f}")
print(classification_report(y_test, rf_preds, target_names=class_names, zero_division=0))

save_confusion_matrix(y_test,rf_preds,"Random Forest")

joblib.dump(rf, os.path.join(model, 'nids_model_rf.pkl')) 
print(f"Model saved in {model}/nids_model_rf.pkl\n")


feat_names = joblib.load(os.path.join(model,'feature_names.pkl'))
importance = rf.feature_importances_
top_idx = np.argsort(importance)[-20:][::-1]
fig, ax = plt.subplots(figsize=(10,6))
ax.barh([feat_names[i] for i in top_idx], importance[top_idx], color='steelblue')
ax.set_xlabel("Importance")
ax.set_title("Important features")
ax.invert_yaxis()
plt.tight_layout()
fi_path = os.path.join(model,'feature_importance.png')
plt.savefig(fi_path, dpi=150)
plt.close()
print(f"Feature importance plot saved in {fi_path}\n")

print("Model Comparison Summary:")
print(f"{'Model':<20}{'Accuracy':>10}{'Weighted F1':>13}")
print(f"{'Neural Network':<20}{nn_acc:>10.4f}{nn_f1:>13.4f}") 
print(f"{'Random Forest':<20}{rf_acc:>10.4f}{rf_f1:>13.4f}")
print("\n Training Complete.")
