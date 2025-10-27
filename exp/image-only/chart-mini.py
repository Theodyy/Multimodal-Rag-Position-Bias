import json
import os
import argparse
import random
from datasets import load_dataset
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from openmatch.generation_utils import preprocess_text, is_numeric_data, is_within_5_percent
import torch
import csv
import time

def load_qrels(qrels_file):
    qrels = {}
    with open(qrels_file) as f:
        tsvreader = csv.DictReader(f, delimiter="\t")
        for row in tsvreader:
            qid = row["query-id"]
            pid = row["corpus-id"]
            rel = int(row["score"])
            if qid in qrels:
                qrels[qid][pid] = rel
            else:
                qrels[qid] = {pid: rel}
    return qrels

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, choices=['chart'], required=True)
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

def save_wrong_images(output_dir, position, qid, doc_order, corpus):
    # 创建错误图片保存目录
    image_output_dir = os.path.join(output_dir, f'wrong_images_position_{position}')
    os.makedirs(image_output_dir, exist_ok=True)

    # 保存qid对应的图片
    if qid in corpus:
        image = corpus[qid]
        image_save_path = os.path.join(image_output_dir, f"wrong_image_{qid}.png")
        image.save(image_save_path)

    # 保存docid对应的图片
    for doc in doc_order:
        if doc in corpus:
            image = corpus[doc]
            image_save_path = os.path.join(image_output_dir, f"wrong_image_{qid}_{doc}.png")
            image.save(image_save_path)

def main():
    start_time=time.time()
    args = parse_args()
    dataset_name = args.dataset_name
    output_dir = args.output_dir

    # 加载语料库   ,请注意修改路径
    corpus_ds = load_dataset(f"../data/chart/cropus", split="train")
    corpus = {item['corpus-id']: item['image'].convert('RGB') for item in corpus_ds}
    print("Loading dataset (corpus)")

    # 加载qrels文件,请注意修改路径
    qrels_file_path = '../data/chart/qrels/qrels_chartqa-eval-qrels.tsv'
    qrels = load_qrels(qrels_file_path)

    # 加载查询,请注意修改路径
    queries_ds = load_dataset(f"../data/chart/queries", split="train")
    print("Loading dataset (queries)")

    # 加载模型
    model_path = '../data/MiniCPM-V-2_6'  # 替换为实际路径
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = model.eval().cuda()
    print("Loading model over")

    # 初始化结果记录
    position_results = {1:[],2:[],3:[],4:[],5:[]}  
    position_history = {1:[],2:[],3:[],4:[],5:[]}  
    position_wrong_answers = {1:[],2:[],3:[],4:[],5:[]} 

    for example in queries_ds:
        qid = example['query-id']
        query = example['query']
        answer = example['answer']

        if qid not in qrels:
            continue  # 如果没有正样本，跳过

        # 获取正样本文档
        positive_doc = qid.rsplit('-', 1)[0]

        # 随机选择两个其他文档
        all_docs = list(corpus.keys())
        other_docs = random.sample([doc for doc in all_docs if doc != positive_doc], 4)

        # 构造三种文档顺序
        doc_orders = [
            [positive_doc] + other_docs, 
            other_docs[:1] + [positive_doc] + other_docs[1:],
            other_docs[:2] + [positive_doc] + other_docs[2:],
            other_docs[:3] + [positive_doc] + other_docs[3:],
            other_docs + [positive_doc] 
        ]

        for order_idx, doc_order in enumerate(doc_orders):
            image_list = [corpus[doc] for doc in doc_order]

            # 构造输入
            input = [{'role': 'user', 'content': f"Answer the question using a single word or phrase.\nQuestion:{query}\nAnswer:"}]

            # 调用模型生成答案
            input = [{'role': 'user', 'content': image_list + [input[0]['content']]}]
            with torch.no_grad():
                responds = model.chat(
                    image=None,
                    msgs=input,
                    tokenizer=tokenizer,
                    sampling=False,
                    max_new_tokens=20
                )

            # 预处理答案
            responds = preprocess_text(responds)
            answer = preprocess_text(answer)
            if '%' in responds and '%' not in answer:
                responds = responds.replace('%', '')
            if '%' not in responds and '%' in answer:
                answer = answer.replace('%', '')

            # 判断答案是否正确
            is_correct = (responds == answer) or (
                is_numeric_data(responds) and is_numeric_data(answer) and answer != '0' and is_within_5_percent(responds, answer)
            )

            # 记录结果
            position = order_idx + 1
            position_results[position].append(is_correct)

            # 记录历史数据
            history_data = {
                'qid': qid,
                'query': query,
                'original_answer': answer,
                'doc_order': doc_order,
                'position': position,
                'generated_answer': responds,
                'is_correct': is_correct
            }
            position_history[position].append(history_data)

            # 记录错误答案并保存错误图片
            if not is_correct:
                wrong_answer_info = {
                    'query': query,
                    'generated_answer': responds,
                    'correct_answer': answer,
                    'qid': qid,
                    'doc_order': doc_order,
                    'position': position
                }
                position_wrong_answers[position].append(wrong_answer_info)

                # 保存错误图片
                save_wrong_images(output_dir, position, qid, doc_order, corpus)

    # 计算每个位置的准确率
    for position in [1, 2, 3,4,5]:
        correct_count = sum(position_results[position])
        total_count = len(position_results[position])
        accuracy = correct_count / total_count if total_count > 0 else 0
        print(f"Position {position} - Accuracy: {accuracy:.4f}")

        # 保存结果
        save_results(
            output_dir,
            f"position_{position}",
            {'accuracy': accuracy, 'correct_count': correct_count, 'total_count': total_count},
            position_history[position],
            position_wrong_answers[position]
        )
    # 记录结束时间
    end_time = time.time()
    # 计算运行时长
    elapsed_time = end_time - start_time
    print(f"代码运行时长: {elapsed_time:.2f} 秒")

if __name__ == '__main__':
    main()