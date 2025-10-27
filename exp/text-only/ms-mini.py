import json
import os
import argparse
import random
import pandas as pd
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    return parser.parse_args()

def save_results(output_dir, prefix, results, history, wrong_answers):
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"{prefix}_results.json")
    history_path = os.path.join(output_dir, f"{prefix}_history.json")
    wrong_answers_path = os.path.join(output_dir, f"{prefix}_wrong_answers.json")

    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    with open(wrong_answers_path, 'w') as f:
        json.dump(wrong_answers, f, indent=4)
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
def main():
    start_time = time.time()
    args = parse_args()
    output_dir = args.output_dir
    device_map = 'auto'

    # 加载ms_marco数据集,请注意修改路径
    data_path = '../data/ms_marco/test-00000-of-00001.parquet'
    data = pd.read_parquet(data_path).head(1000)
    print(f"Loaded ms_marco dataset: {len(data)} rows")

    # 填写自己的模型路径-加载模型
    model_path = '../data/MiniCPM-V-2_6' 
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    model = model.eval()
    print("Model loaded.")

    # 加载语义相似度模型
    st_model = SentenceTransformer('../data/all-MiniLM-L6-v2')
    print("SentenceTransformer loaded.")

    # 初始化结果记录,k=3
    position_results = {1: [], 2: [], 3: []}  # 正确段落在第1、2、3位
    # position_history = {1: [], 2: [], 3: []}
    # position_wrong_answers = {1: [], 2: [], 3: []}
    # 如果需要测试5个位置，请取消注释下面的代码
    # position_results = {1:[],2:[],3:[],4:[],5:[]}  
    # position_history = {1:[],2:[],3:[],4:[],5:[]}  
    # position_wrong_answers = {1:[],2:[],3:[],4:[],5:[]} 


    for idx, row in data.iterrows():
        answers = row['answers']  # list
        query = row['query']
        passages = row['passages']  # dict
        is_selected = passages['is_selected']
        passage_texts = passages['passage_text']

        # 找到正确段落和错误段落
        correct_indices = [i for i, sel in enumerate(is_selected) if sel == 1]
        wrong_indices = [i for i, sel in enumerate(is_selected) if sel == 0]

        if not correct_indices or len(wrong_indices) < 2:
            continue  # 跳过不符合条件的数据

        correct_idx = correct_indices[0]  # 只用第一个正确段落
        wrong_idx_sample = random.sample(wrong_indices, 2)
        selected_passages = [passage_texts[correct_idx]] + [passage_texts[i] for i in wrong_idx_sample]

        # 构造三种顺序
        orders = [
            [0, 1, 2],  # 正确段落在第1位
            [1, 0, 2],  # 正确段落在第2位
            [2, 1, 0],  # 正确段落在第3位
        ]
        # 如果需要测试5个位置，请取消注释下面的代码
        # other_docs = random.sample([doc for doc in all_docs if doc != positive_doc], 4)
        # doc_orders = [
        #     [positive_doc] + other_docs, 
        #     other_docs[:1] + [positive_doc] + other_docs[1:],
        #     other_docs[:2] + [positive_doc] + other_docs[2:],
        #     other_docs[:3] + [positive_doc] + other_docs[3:],
        #     other_docs + [positive_doc] 
        # ]         
        for pos, order in enumerate(orders, 1):
            passages_ordered = [selected_passages[i] for i in order]
            prompt = (
                "Answer the question using a single word or phrase. Your answer should be concise and directly address the question.\n"
                f"Question: {query}\nPassages:\n"
            )
            for i, passage in enumerate(passages_ordered):
                prompt += f"{passage}\n"
            prompt += "Answer:"

            input_data = [{'role': 'user', 'content': prompt}]
            with torch.no_grad():
                responds = model.chat(
                    image=None,
                    msgs=input_data,
                    tokenizer=tokenizer,
                    sampling=False,
                    max_new_tokens=50
                )
            if isinstance(responds, dict) and 'response' in responds:
                model_answer = responds['response']
            else:
                model_answer = str(responds)
            print(f"QID {row['query_id']} | Position {pos} | Model Answer: {model_answer}")

            # 语义相似度判断
            answer_scores = [util.pytorch_cos_sim(
                st_model.encode(model_answer, convert_to_tensor=True),
                st_model.encode(ans, convert_to_tensor=True)
            ).item() for ans in answers if ans.strip()]
            max_score = max(answer_scores) if answer_scores else 0.0
            is_correct = max_score > 0.7  # 阈值可调

            # 记录结果
            position_results[pos].append(is_correct)
            history_data = {
                'qid': row['query_id'],
                'query': query,
                'passages': passages_ordered,
                'position': pos,
                'generated_answer': model_answer,
                'is_correct': is_correct,
                'max_score': max_score,
                'answers': answers
            }
            position_history[pos].append(history_data)

            if not is_correct:
                wrong_answer_info = {
                    'qid': row['query_id'],
                    'query': query,
                    'passages': passages_ordered,
                    'position': pos,
                    'generated_answer': model_answer,
                    'max_score': max_score,
                    'answers': answers
                }
                position_wrong_answers[pos].append(wrong_answer_info)

    # 计算每个位置的准确率
    for position in [1, 2, 3]:
        correct_count = sum(position_results[position])
        total_count = len(position_results[position])
        accuracy = correct_count / total_count if total_count > 0 else 0
        print(f"Position {position} - Accuracy: {accuracy:.4f}")
        save_results(
            output_dir,
            f"position_{position}",
            {'accuracy': accuracy, 'correct_count': correct_count, 'total_count': total_count},
            position_history[position],
            position_wrong_answers[position]
        )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"代码运行时长: {elapsed_time:.2f} 秒")
    with open(os.path.join(output_dir, 'run_time.txt'), 'w') as f:
        f.write(f"Elapsed time: {elapsed_time:.2f} seconds\n")

if __name__ == '__main__':
    main()