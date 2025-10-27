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
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import numpy as np
import re
from openai import OpenAI
import requests
import re
import io

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

def upload_to_imgur(image, client_id, max_retries=3):
    """上传图片到Imgur并返回公开URL"""
    headers = {"Authorization": f"Client-ID {client_id}"}
    
    # 将PIL Image转换为字节流
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.imgur.com/3/image",
                headers=headers,
                files={"image": img_byte_arr}
            )
            if response.status_code == 200:
                return response.json()["data"]["link"]
            else:
                print(f"上传失败，重试 {attempt + 1}/{max_retries}。状态码: {response.status_code}")
        except Exception as e:
            print(f"上传异常: {e}")
    
    return None  # 返回None而不是抛出异常
    

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
def main():
    start_time=time.time()
    args = parse_args()
    output_dir = args.output_dir
    device_map = 'auto'

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

    # 填写自己的imgur_client_id
    imgur_client_id = ""

    # 填写自己的API_KEY-加载大模型
    API_KEY = ""
    # 创建客户端实例
    client = OpenAI(api_key=API_KEY)
    print("Loading model over")

    # 初始化结果记录,k=3
    position_results = {1:[],2:[],3:[]}  
    position_history = {1:[],2:[],3:[]}  
    position_wrong_answers = {1:[],2:[],3:[]} 
    # 如果需要测试5个位置，请取消注释下面的代码
    # position_results = {1:[],2:[],3:[],4:[],5:[]}  
    # position_history = {1:[],2:[],3:[],4:[],5:[]}  
    # position_wrong_answers = {1:[],2:[],3:[],4:[],5:[]} 

    # i=0
    for example in queries_ds:
        # i+=1
        # if i>1:
        #     break
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
            prompt = f"Answer the question using a single word or phrase.  \nQuestion:{query}\nAnswer:"
            image_url_list = []
            upload_success = True
            for i, img in enumerate(image_list):
                try:
                    image_url = upload_to_imgur(
                        image=img,
                        client_id=imgur_client_id
                    )
                    if image_url is None:
                        print(f"图片 {i+1} 上传失败，跳过当前样本")
                        upload_success = False
                        break
                    image_url_list.append(image_url)
                except Exception as e:
                    print(f"图片 {i+1} 上传异常: {str(e)}")
                    upload_success = False
                    break
            
            if not upload_success:
                continue  # 跳过当前样本，处理下一个

            response = client.chat.completions.create(
                model="gpt-4o",
                messages = [
                    {
                        "role": "user",
                        "content": [
                            *[{"type": "image_url", "image_url": {"url": img_url}} for img_url in image_url_list],
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
                temperature=0.3
            )
            responds = response.choices[0].message.content.strip()

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