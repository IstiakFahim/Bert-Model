import pandas as pd
import json

# ===== prompt 模板（映射明文写死） =====
PROMPT = (
    "You are a multi-emotion classification model.\n"
    "Each text may contain one or more emotions.\n"
    "Use the following mapping and output only the corresponding numbers.\n\n"
    '{"admiration": 0, "amusement": 1, "anger": 2, "annoyance": 3, "approval": 4, '
    '"caring": 5, "confusion": 6, "curiosity": 7, "desire": 8, "disappointment": 9, '
    '"disapproval": 10, "disgust": 11, "embarrassment": 12, "excitement": 13, "fear": 14, '
    '"gratitude": 15, "grief": 16, "joy": 17, "love": 18, "nervousness": 19, "neutral": 20, '
    '"optimism": 21, "pride": 22, "realization": 23, "relief": 24, "remorse": 25, '
    '"sadness": 26, "surprise": 27}\n\n'
    "Output format example:\n"
    'Text: \"I finally finished my project and it feels amazing!\"\n'
    "→ Output: 13,17,22\n\n"
    "Now classify the following text and output only the numbers (comma-separated, no extra words)."
)

# ===== 数据加载与合并 =====
def load_and_merge(path):
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str).str.strip()
    df["emotion"] = df["emotion"].astype(str).str.strip()
    df = df.groupby("text")["emotion"].apply(lambda x: list(set(x))).reset_index()
    return df

# ===== 生成 JSON 文件 =====
def make_json(input_path, output_path):
    df = load_and_merge(input_path)
    data = []
    # 标签映射表（同样写死，避免依赖）
    label2id = {
        "admiration": 0, "amusement": 1, "anger": 2, "annoyance": 3, "approval": 4,
        "caring": 5, "confusion": 6, "curiosity": 7, "desire": 8, "disappointment": 9,
        "disapproval": 10, "disgust": 11, "embarrassment": 12, "excitement": 13, "fear": 14,
        "gratitude": 15, "grief": 16, "joy": 17, "love": 18, "nervousness": 19, "neutral": 20,
        "optimism": 21, "pride": 22, "realization": 23, "relief": 24, "remorse": 25,
        "sadness": 26, "surprise": 27
    }

    for _, row in df.iterrows():
        label_ids = [label2id[e] for e in row["emotion"] if e in label2id]
        label_str = ",".join(map(str, sorted(label_ids)))
        data.append({
            "instruction": PROMPT,
            "input": row["text"],
            "output": label_str
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} samples → {output_path}")

# ===== 主执行入口 =====
if __name__ == "__main__":
    make_json("./train.csv", "emotion_train.json")
    make_json("./validation.csv", "emotion_validation.json")
    make_json("./test.csv", "emotion_test.json")
