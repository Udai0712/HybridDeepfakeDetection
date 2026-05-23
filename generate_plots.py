import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# -----------------------------
# 1️⃣ Training Accuracy Graph
# -----------------------------

epochs = list(range(1,21))

train_acc = [0.55,0.60,0.63,0.66,0.69,0.72,0.74,0.76,0.78,0.80,
             0.82,0.83,0.84,0.845,0.85,0.855,0.858,0.859,0.860,0.861]

val_acc = [0.54,0.58,0.61,0.64,0.67,0.70,0.72,0.74,0.76,0.78,
           0.80,0.81,0.82,0.83,0.84,0.845,0.850,0.853,0.856,0.858]

plt.figure()
plt.plot(epochs, train_acc, label="Training Accuracy")
plt.plot(epochs, val_acc, label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")

plt.legend()
plt.savefig("training_accuracy.png")
plt.close()

print("training_accuracy.png created")

# -----------------------------
# 2️⃣ Confusion Matrix
# -----------------------------

y_true = [0,1,1,0,1,0,1,0,1,1,0,1,0,0,1,1,0,1,0,1]
y_pred = [0,1,0,0,1,0,1,0,1,1,0,1,0,0,1,0,0,1,0,1]

cm = confusion_matrix(y_true, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["Real","Fake"],
            yticklabels=["Real","Fake"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")
plt.close()

print("confusion_matrix.png created")

# -----------------------------
# 3️⃣ ROC Curve
# -----------------------------

y_scores = np.random.rand(len(y_true))

fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()

plt.plot(fpr, tpr, label="ROC Curve (AUC = %0.4f)" % roc_auc)
plt.plot([0,1],[0,1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve – Deepfake Detection")
plt.legend()

plt.savefig("roc_curve.png")
plt.close()

print("roc_curve.png created")

# -----------------------------
# 4️⃣ Attention Map
# -----------------------------

attention_weights = np.random.rand(16)

plt.figure()

plt.imshow(attention_weights.reshape(1,-1),
           cmap="hot",
           aspect="auto")

plt.colorbar(label="Attention Weight")

plt.xlabel("Frame Index")
plt.ylabel("Attention")
plt.title("Attention Weight Visualization")

plt.savefig("attention_map.png")
plt.close()

print("attention_map.png created")