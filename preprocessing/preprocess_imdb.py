# preprocess_imdb
# IMDB 데이터 전처리

import os
import pandas as pd
from datasets import load_dataset

def convert_imdb_to_tsv():
    print("Loading IMDB Datasets from Hugging Face...")

    # 데이터셋 로드
    dataset=load_dataset("imdb")

    # train 전처리
    train_df=pd.DataFrame({
        'id':range(len(dataset['train'])),
        'sentence':dataset['train']['text'],
        'sentiment': dataset['train']['label']
    })

    train_df['sentiment']=train_df['sentiment'].astype(str)

    # dev 전처리
    dev_df=pd.DataFrame({
        'id':range(len(dataset['test'])),
        'sentence':dataset['test']['text'],
        'sentiment': dataset['test']['label']
    })

    dev_df['sentiment']=dev_df['sentiment'].astype(str)

    # TSV 저장
    train_df.to_csv('data/imdb-train.csv',sep='\t',index=False)
    dev_df.to_csv('data/imdb-dev.csv',sep='\t',index=False)

    print("Complete Preprocessing IMDB!!")

if __name__=="__main__":
    convert_imdb_to_tsv()