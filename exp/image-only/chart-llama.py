import json
import os
import argparse
import random
from datasets import load_dataset
from PIL import Image
from openmatch.generation_utils import preprocess_text, is_numeric_data, is_within_5_percent
import torch
import csv
import time
from PIL import Image, ImageDraw, ImageFont
from transformers import MllamaForConditionalGeneration, AutoProcessor, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import numpy as np
import re

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
    parser.add_argument('--output_dir', type=str, required=True)
    return parser.parse_args()

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def save_results(output_dir, prefix, results, history, wrong_answers):
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"{prefix}_results.json")
    history_path = os.path.join(output_dir, f"{prefix}_history.json")
    wrong_answers_path = os.path.join(output_dir, f"{prefix}_wrong_answers.json")

    # 转换数据为可序列化格式
    results = convert_to_serializable(results)
    history = convert_to_serializable(history)
    wrong_answers = convert_to_serializable(wrong_answers)

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
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
def main():
    start_time=time.time()
    args = parse_args()
    output_dir = args.output_dir
    device_map = 'cuda:7'

    # 加载语料库，请注意修改路径
    corpus_ds = load_dataset(f"../data/chart/cropus", split="train")
    corpus = {item['corpus-id']: item['image'].convert('RGB') for item in corpus_ds}
    print("Loading dataset (corpus)")

    # 加载qrels文件，请注意修改路径
    qrels_file_path = '../data/chart/qrels/qrels_chartqa-eval-qrels.tsv'
    qrels = load_qrels(qrels_file_path)

    # 加载查询，请注意修改路径
    queries_ds = load_dataset(f"../data/chart/queries", split="train")
    print("Loading dataset (queries)")

    # 填写自己的模型路径-加载模型
    model_name = "/Data2/LLMs/Llama-3.2-11B-Vision-Instruct"
    model = MllamaForConditionalGeneration.from_pretrained(  
        model_name,  
        torch_dtype=torch.bfloat16,
        device_map=device_map, 
        attn_implementation="eager"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    print("Loading model over")

    # 初始化结果记录,k=3
    position_results = {1:[],2:[],3:[]}  
    position_history = {1:[],2:[],3:[]}  
    position_wrong_answers = {1:[],2:[],3:[]} 
    # 如果需要测试5个位置，请取消注释下面的代码
    # position_results = {1:[],2:[],3:[],4:[],5:[]}  
    # position_history = {1:[],2:[],3:[],4:[],5:[]}  
    # position_wrong_answers = {1:[],2:[],3:[],4:[],5:[]} 

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
        other_docs = random.sample([doc for doc in all_docs if doc != positive_doc], 2)
        # 如果需要测试5个位置，请取消注释下面的代码
        # other_docs = random.sample([doc for doc in all_docs if doc != positive_doc], 4)
        # 构造三种文档顺序
        doc_orders = [
            [positive_doc] + other_docs, 
            other_docs[:1] + [positive_doc] + other_docs[1:],
            other_docs + [positive_doc] 
        ]
        # 如果需要测试5个位置，请取消注释下面的代码
        # doc_orders = [
        #     [positive_doc] + other_docs, 
        #     other_docs[:1] + [positive_doc] + other_docs[1:],
        #     other_docs[:2] + [positive_doc] + other_docs[2:],
        #     other_docs[:3] + [positive_doc] + other_docs[3:],
        #     other_docs + [positive_doc] 
        # ]         
        for order_idx, doc_order in enumerate(doc_orders):
            image_list = [corpus[doc] for doc in doc_order]

            # 构造输入
            prompt = f"ATTENTION!!!!!!: A single word or phrase!!! \nAnswer the question using a single word or phrase.  \nQuestion:{query}\nAnswer:"
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in image_list],
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            input_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                images=image_list,
                text=input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device)
            output_ids = model.generate(**inputs, max_new_tokens=128)
            # 关键修改：截取新生成的token部分
            input_length = inputs.input_ids.shape[1]  # 获取输入序列长度
            generated_ids = output_ids[:, input_length:]  # 只保留新生成的部分
            responds = processor.batch_decode(
                generated_ids,  # 使用batch_decode而不是decode
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            responds = responds[0].strip()  # 获取第一个生成的回答

            # 预处理答案
            responds = preprocess_text(responds)
            print(responds)
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
    # 如果需要测试5个位置，请取消注释下面的代码
    # for position in [1, 2, 3, 4, 5]:
    for position in [1, 2, 3]:
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