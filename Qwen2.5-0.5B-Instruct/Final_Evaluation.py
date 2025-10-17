#!/usr/bin/env python3
import json
import numpy as np
from collections import Counter
from sklearn.metrics import (
    f1_score, hamming_loss, accuracy_score, 
    precision_score, recall_score, classification_report
)

# ==================== 配置 ====================
PREDICTIONS_FILE = "/scratch/project_2015499/AC_project/qwen25_emotion_sft_test_results/generated_predictions.jsonl"

EMOTION_MAP = {
    0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance",
    4: "approval", 5: "caring", 6: "confusion", 7: "curiosity",
    8: "desire", 9: "disappointment", 10: "disapproval", 11: "disgust",
    12: "embarrassment", 13: "excitement", 14: "fear", 15: "gratitude",
    16: "grief", 17: "joy", 18: "love", 19: "nervousness",
    20: "neutral", 21: "optimism", 22: "pride", 23: "realization",
    24: "relief", 25: "remorse", 26: "sadness", 27: "surprise"
}

NUM_LABELS = 28

# ==================== 工具函数 ====================
def parse_emotions(emotion_str):
    """解析情感字符串"""
    emotion_str = emotion_str.strip().replace('\n', '').replace(' ', '')
    if not emotion_str:
        return []
    try:
        emotions = [int(x.strip()) for x in emotion_str.split(',') if x.strip()]
        return emotions
    except:
        return []

def emotions_to_multilabel(emotion_ids, num_labels=NUM_LABELS):
    """将情感ID列表转为多标签向量"""
    vec = np.zeros(num_labels, dtype=int)
    for eid in emotion_ids:
        if 0 <= eid < num_labels:
            vec[eid] = 1
    return vec

def jaccard_score_example_based(y_true, y_pred):
    """样本级Jaccard相似度"""
    inter = (y_true & y_pred).sum(axis=1)
    union = (y_true | y_pred).sum(axis=1)
    union = np.where(union == 0, 1, union)
    return (inter / union).mean()

# ==================== 读取数据 ====================
print("📖 Loading predictions...")

predictions_list = []
labels_list = []

with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        pred_emotions = parse_emotions(data['predict'])
        true_emotions = parse_emotions(data['label'])
        
        predictions_list.append(emotions_to_multilabel(pred_emotions))
        labels_list.append(emotions_to_multilabel(true_emotions))

y_pred = np.array(predictions_list)
y_true = np.array(labels_list)

print(f" Loaded {len(y_true)} samples")

# ==================== 计算指标 ====================
print("\n" + "="*70)
print("📊 MULTI-LABEL CLASSIFICATION METRICS")
print("="*70)

# 1. F1 Scores
micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print("\n F1 Scores:")
print(f"  Micro F1    : {micro_f1:.4f}  ")
print(f"  Macro F1    : {macro_f1:.4f}  ")
print(f"  Weighted F1 : {weighted_f1:.4f}  ")

# 2. Precision & Recall
micro_precision = precision_score(y_true, y_pred, average="micro", zero_division=0)
micro_recall = recall_score(y_true, y_pred, average="micro", zero_division=0)
macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

print("\n Precision & Recall:")
print(f"  Micro Precision: {micro_precision:.4f}")
print(f"  Micro Recall   : {micro_recall:.4f}")
print(f"  Macro Precision: {macro_precision:.4f}")
print(f"  Macro Recall   : {macro_recall:.4f}")

# 3. Jaccard Score
jaccard = jaccard_score_example_based(y_true, y_pred)
print(f"\n Jaccard Score (Example-based): {jaccard:.4f}")

# 4. Hamming Loss
h_loss = hamming_loss(y_true, y_pred)
print(f"\n Hamming Loss: {h_loss:.4f}  ")

# 5. Subset Accuracy (严格匹配)
subset_acc = accuracy_score(y_true, y_pred)
print(f"\n Subset Accuracy: {subset_acc:.4f}  ")

# 6. 样本级准确率(至少预测对一个标签)
correct_at_least_one = 0
for yt, yp in zip(y_true, y_pred):
    if np.any(yt & yp):  # 至少有一个标签预测对
        correct_at_least_one += 1
partial_acc = correct_at_least_one / len(y_true)
print(f"\n Partial Match Accuracy: {partial_acc:.4f}  ")

# ==================== Per-Class 指标 ====================
print("\n" + "="*70)
print(" PER-CLASS PERFORMANCE (Top 15 by Support)")
print("="*70)

per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
per_class_support = y_true.sum(axis=0)

# 按support排序
class_stats = []
for i in range(NUM_LABELS):
    class_stats.append({
        'id': i,
        'name': EMOTION_MAP[i],
        'f1': per_class_f1[i],
        'precision': per_class_precision[i],
        'recall': per_class_recall[i],
        'support': int(per_class_support[i])
    })

class_stats_sorted = sorted(class_stats, key=lambda x: x['support'], reverse=True)

print(f"\n{'ID':<3} {'Emotion':<15} {'F1':<7} {'Prec':<7} {'Rec':<7} {'Support':<8}")
print("-" * 70)
for stat in class_stats_sorted[:15]:
    print(f"{stat['id']:<3} {stat['name']:<15} "
          f"{stat['f1']:<7.3f} {stat['precision']:<7.3f} "
          f"{stat['recall']:<7.3f} {stat['support']:<8}")

# ==================== 多标签特定分析 ====================
print("\n" + "="*70)
print(" MULTI-LABEL SPECIFIC ANALYSIS")
print("="*70)

# 真实的多标签样本
true_multi = (y_true.sum(axis=1) > 1)
pred_multi = (y_pred.sum(axis=1) > 1)

print(f"\n真实多标签样本数: {true_multi.sum()} / {len(y_true)} ({true_multi.sum()/len(y_true)*100:.2f}%)")
print(f"预测多标签样本数: {pred_multi.sum()} / {len(y_true)} ({pred_multi.sum()/len(y_true)*100:.2f}%)")

# 只在多标签样本上计算指标
if true_multi.sum() > 0:
    multi_micro_f1 = f1_score(y_true[true_multi], y_pred[true_multi], average="micro", zero_division=0)
    multi_jaccard = jaccard_score_example_based(y_true[true_multi], y_pred[true_multi])
    print(f"\n📌 多标签样本的 Micro F1: {multi_micro_f1:.4f}")
    print(f"📌 多标签样本的 Jaccard: {multi_jaccard:.4f}")

# ==================== 保存结果 ====================
results = {
    "micro_f1": float(micro_f1),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "micro_precision": float(micro_precision),
    "micro_recall": float(micro_recall),
    "macro_precision": float(macro_precision),
    "macro_recall": float(macro_recall),
    "jaccard_score": float(jaccard),
    "hamming_loss": float(h_loss),
    "subset_accuracy": float(subset_acc),
    "partial_match_accuracy": float(partial_acc),
    "per_class_metrics": class_stats_sorted
}

with open("comprehensive_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "="*70)
print(" Complete metrics saved to comprehensive_metrics.json")
print("="*70)