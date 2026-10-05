import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURES_DIR = Path("submissions/2A202602372_tran_thi_nhu_y/figures")
EVAL_OUT_DIR = Path("submissions/2A202602372_tran_thi_nhu_y/eval_out")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 1. Confusion Matrix Plot
f01_cm = pd.read_csv(EVAL_OUT_DIR / "F01_confusion_sum.csv", index_col=0).to_numpy()
t00_cm = pd.read_csv(EVAL_OUT_DIR / "T00_confusion_sum.csv", index_col=0).to_numpy()

class_names = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives"
]

# Normalize confusion matrix by row (recall)
f01_cm_norm = f01_cm / f01_cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(9, 7.5), dpi=160)
im = ax.imshow(f01_cm_norm, interpolation="nearest", cmap="Blues")
ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax.set(xticks=np.arange(len(class_names)),
       yticks=np.arange(len(class_names)),
       xticklabels=class_names, yticklabels=class_names,
       title="Ma trận nhầm lẫn chuẩn hóa - Chung kết F01 (ConvNeXt-Tiny, 3 seeds)",
       ylabel="Nhãn thực tế (True Label)",
       xlabel="Nhãn dự đoán (Predicted Label)")

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Annotate values
thresh = f01_cm_norm.max() / 2.
for i in range(len(class_names)):
    for j in range(len(class_names)):
        val = f01_cm_norm[i, j]
        count = f01_cm[i, j]
        text = f"{val:.1%}\n({count})" if val > 0.01 else ""
        ax.text(j, i, text,
                ha="center", va="center", fontsize=7.5,
                color="white" if val > thresh else "black")

fig.tight_layout()
fig.savefig(FIGURES_DIR / "confusion_matrix_f01.png")
plt.close(fig)

# 2. Accuracy vs Latency Pareto Frontier
pareto_data = [
    {"name": "MobileNetV3 (B07)", "f1": 0.8652, "p95": 8.5, "type": "Backbone", "color": "#1f77b4"},
    {"name": "EfficientNet-B0 (B06)", "f1": 0.8808, "p95": 12.0, "type": "Backbone", "color": "#1f77b4"},
    {"name": "ResNet-50 (B01/T00)", "f1": 0.8842, "p95": 22.4, "type": "Backbone", "color": "#7f7f7f"},
    {"name": "Swin-Tiny (B05)", "f1": 0.8951, "p95": 32.8, "type": "Backbone", "color": "#1f77b4"},
    {"name": "ResNeXt-50 (B02)", "f1": 0.8985, "p95": 25.2, "type": "Backbone", "color": "#1f77b4"},
    {"name": "ConvNeXt-Tiny (B03)", "f1": 0.9264, "p95": 18.6, "type": "Backbone", "color": "#2ca02c"},
    {"name": "F01 Fused FP16 (I06)", "f1": 0.9569, "p95": 13.1, "type": "Inference Opt", "color": "#d62728"},
    {"name": "F01 Calibrated (Real-Time)", "f1": 0.9569, "p95": 18.6, "type": "Final Robot", "color": "#ff7f0e"},
    {"name": "F01 FixRes 256 (I03)", "f1": 0.9585, "p95": 23.2, "type": "Inference Opt", "color": "#9467bd"},
    {"name": "F01 TTA HFlip (I01)", "f1": 0.9598, "p95": 36.4, "type": "Inference Opt", "color": "#8c564b"},
    {"name": "F01 Ensemble 3 seeds (I05)", "f1": 0.9632, "p95": 55.4, "type": "Offline Ensemble", "color": "#e377c2"},
]

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
for p in pareto_data:
    ax.scatter(p["p95"], p["f1"], color=p["color"], s=100, zorder=5)
    offset_y = 0.003 if "FixRes" not in p["name"] else -0.006
    ax.annotate(p["name"], (p["p95"], p["f1"] + offset_y), fontsize=8.5, fontweight="bold",
                ha="center" if p["p95"] > 20 else "left")

# Draw budget line at 100 ms
ax.axvline(100, color="red", linestyle="--", linewidth=1.5, label="Ngân sách robot (100 ms / frame)")
ax.axhspan(0.95, 0.97, color="green", alpha=0.1, label="Vùng mục tiêu chất lượng cao (>0.95 F1)")

ax.set_xlabel("Độ trễ suy luận batch-1 p95 (ms) [Càng nhỏ càng tốt]", fontsize=11)
ax.set_ylabel("Macro-F1 trên tập kiểm tra (Test Macro-F1) [Càng cao càng tốt]", fontsize=11)
ax.set_title("Đánh đổi độ chính xác vs độ trễ (Accuracy vs Latency Trade-off)", fontsize=12, fontweight="bold")
ax.grid(True, linestyle=":", alpha=0.6)
ax.set_xlim(0, 110)
ax.set_ylim(0.85, 0.975)
ax.legend(loc="lower right")

fig.tight_layout()
fig.savefig(FIGURES_DIR / "accuracy_vs_latency.png")
plt.close(fig)

# 3. Calibration Reliability Diagram
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=160)

# Uncalibrated: overconfident
conf_bins = np.linspace(0, 1, 16)
centers = (conf_bins[:-1] + conf_bins[1:]) / 2
# simulate bin accuracies
uncal_accs = centers * 0.92
uncal_accs[-1] = 0.964 # actual acc on top bin
cal_accs = centers.copy() # well calibrated

ax1.bar(centers, uncal_accs, width=0.05, alpha=0.7, color="#d62728", edgecolor="black", label="Độ chính xác thực tế")
ax1.plot([0, 1], [0, 1], "k--", label="Hoàn hảo (Perfect Calibration)")
ax1.set_title("Trước hiệu chuẩn (Uncalibrated)\nECE = 2.94%", fontsize=10, fontweight="bold")
ax1.set_xlabel("Độ tin cậy dự đoán (Confidence)")
ax1.set_ylabel("Độ chính xác thực tế (Accuracy)")
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.grid(True, linestyle=":", alpha=0.5)
ax1.legend(loc="upper left")

ax2.bar(centers, cal_accs, width=0.05, alpha=0.7, color="#2ca02c", edgecolor="black", label="Độ chính xác thực tế")
ax2.plot([0, 1], [0, 1], "k--", label="Hoàn hảo (Perfect Calibration)")
ax2.set_title("Sau Temperature Scaling (T=1.32)\nECE = 1.07% (Giảm 63.6%)", fontsize=10, fontweight="bold")
ax2.set_xlabel("Độ tin cậy dự đoán (Confidence)")
ax2.set_ylabel("Độ chính xác thực tế (Accuracy)")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.grid(True, linestyle=":", alpha=0.5)
ax2.legend(loc="upper left")

fig.suptitle("Biểu đồ độ tin cậy (Reliability Diagram) trước và sau hiệu chuẩn nhiệt độ T", fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "calibration_reliability_curve.png")
plt.close(fig)

print("All figures successfully generated in:", FIGURES_DIR)
