This project is trained using the LLaMA Factory framework.
To reproduce the results, you must have a basic understanding of LLaMA Factory — this repository does not cover how to use the framework itself...I have no time to write a guideline.

Alternatively, to quickly verify the results, I can invite you to my CSC platform project, where the server logs, files, and timestamped outputs are all available.

1.Get_json.py
Uses the same CSV dataset as BERT and converts it into JSON files for LLM fine-tuning:
emotion_train.json
emotion_validation.json
emotion_test.json
You’ll need to learn how LLaMA Factory organizes its data and how to configure these three JSON files under the data directory.

2.Check_json.py
Since the original CSV dataset uses multiple rows to represent multi-label data, this script checks and merges them into a single entry in JSON format, consolidating all labels under one output field.

3.qwen25_emotion.yaml
Training configuration file using LoRA.

4.qwen25_emotion_SFT_test.yaml
Testing configuration for the LoRA fine-tuned model.
The inference results are saved in the qwen25_emotion_sft_test_results folder, which includes:
all_results.json
generated_predictions.jsonl
predict_results.json
trainer_log.jsonl

5.qwen25_emotion_basemodel_test.yaml
Testing configuration for the base model (without fine-tuning).
The results are saved in the qwen25_emotion_basemodel_test_results folder, which includes:
all_results.json
generated_predictions.jsonl
predict_results.json
trainer_log.jsonl

6.Final_Evaluation.py
The final evaluation script for analyzing the fine-tuned model only.
Key information is printed to the console and also saved to comprehensive_metrics.json.
BECAUSE The base model performs poorly — producing random, garbled, or invalid labels. You can sample its raw outputs in qwen25_emotion_basemodel_test_results/generated_predictions.jsonl to see examples.