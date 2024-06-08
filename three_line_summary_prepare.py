from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from torch import Tensor
import json
import numpy as np
from tqdm import tqdm  # Import tqdm

# ２つの文章の類似度を計算するクラス
class similarityManager():
    def __init__(self) -> None:
        self.emb_tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
        self.emb_model = AutoModel.from_pretrained('intfloat/multilingual-e5-large').to('cuda')  # Move the model to GPU
    
    def average_pool(
        self,
        last_hidden_states: Tensor,
        attention_mask: Tensor
    ) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    # ２つの文章の類似度を計算しそのスコアを返す
    def get_similarity(self, text1, text2):
        tokenizer=self.emb_tokenizer
        model=self.emb_model

        input_texts = [f"query: {text1}", f"passage: {text2}"]
        batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to('cuda')  # Move the input tensors to GPU
        
        outputs = model(**batch_dict)
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = (embeddings[:1] @ embeddings[1:].T) * 100
        scores = np.asarray(scores.tolist())
        
        return scores[0][0]

import json
import csv
import random

MAX_DATA_NUM = 5000 #+ random.randint(0, 100)
sm = similarityManager()

# tsvファイルを読み込む関数
def read_tsv(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        data = list(reader)
    return data

def pre_process_data(data):
    processed_data = []
    
    # Wrap your data iterable with tqdm for a progress bar
    for row in tqdm(data, desc="Processing rows"):
        try:
            row[3]
        except IndexError:
            continue
        
        summaries = [f"{summary}。" for summary in row[:3]]
        summary = ''.join(summaries)
        
        input_text = row[3]

        # Assuming sm.get_similarity is defined elsewhere
        similarity = sm.get_similarity(input_text, summary)
        
        processed_data.append({'input': input_text, 'summary': summary, 'similarity': similarity})
    
    # Print the number of original and filtered data points
    print("original data: ", len(data))
    print("filtered data: ", len(processed_data))
    
    return processed_data

# pre_process_data の結果をフィルタする関数
def filter_data(data):
    filtered_data = []
    
    # Wrap your data iterable with tqdm for a progress bar
    for row in tqdm(data, desc="Processing rows"):
        try:
            row['similarity']
        except IndexError:
            continue
        
        # 文章が長すぎる場合はスキップ
        #if len(row['input']) > 512:
        #    continue

        # 文章が短すぎる場合はスキップ
        if len(row['summary']) < 100 or len(row['input']) < 100:
            continue
        
        # ある程度長い文章のみを要約する
        if len(row['summary']) * 3 > len(row['input']):
            continue
        
        # 長過ぎる(context長に収まらない場合はスキップ
        if len(row['summary']) + len(row['input']) > 1600:
            continue
            
        filtered_data.append(row)
            
    # Print the number of original and filtered data points
    print("original data: ", len(data))
    print("filtered data: ", len(filtered_data))
    
    return filtered_data

# JSON形式で保存する関数
def save_to_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

# スクリプト実行部分
def main():
    # tsvファイルのパスを指定
    tsv_file_path = './ThreeLineSummaryDataset/work/output.tsv'
    # JSONファイルの保存先を指定
    json_prepare_file_path = './dataset/three_line_summary_prepare.json'
    json_file_path = './dataset/three_line_summary.json'
    
    if False:
        # 新しくデータを処理する場合        
        tsv_data = read_tsv(tsv_file_path)
        processed_data = pre_process_data(tsv_data)
    
        # similarity が高い順に並び替える
        processed_data = sorted(processed_data, key=lambda x: x['similarity'], reverse=True)
        
        print("prepared data: ", len(processed_data))
        
        # 処理したデータをJSON形式で保存する
        save_to_json(processed_data, json_prepare_file_path)
    else:
        # 既存のJSONファイルを読み込む場合
        with open(json_prepare_file_path, 'r', encoding='utf-8') as file:
            processed_data = json.load(file)
    
    # フィルタする
    processed_data = filter_data(processed_data)
    
    # 上から MAX_DATA_NUM 件のデータを抽出する
    processed_data = processed_data[:MAX_DATA_NUM]
    
    # processed_data の summary が短い順に並び替える
    #processed_data = sorted(processed_data, key=lambda x: len(x['summary']), reverse=False)
    
    # データを MAX_DATA_NUM になるように均等に間引く
    #processed_data = processed_data[random.randint(0, 100):]
    #processed_data = [processed_data[i] for i in range(0, len(processed_data), len(processed_data) // MAX_DATA_NUM)]
    
    # ランダムな MAX_DATA_NUM 件のデータを抽出する
    #random.shuffle(processed_data)
    #processed_data = processed_data[:MAX_DATA_NUM]
    
    print("processed data: ", len(processed_data))
    
    # 処理したデータをJSON形式で保存する
    save_to_json(processed_data, json_file_path)

if __name__ == '__main__':
    main()
