# 事前に https://github.com/KodairaTomonori/ThreeLineSummaryDataset から ThreeLineSummaryDataset/data/train.csv をダウンロードしておいてください。

from urllib.request import urlopen
from bs4 import BeautifulSoup
from bs4.element import NavigableString
import time

# 収集するニュース記事のインデックス

start_index = 194701 # 開始インデックス
end_index = 213160 # 終了インデックス←ここに取得したい件数を指定します。最初は10件でテストします。

# コンテンツの取得
def get_content(id):
    # サーバに負荷をかけないように10秒スリープ
    time.sleep(2)

    # URL
    URL = 'https://news.livedoor.com/article/detail/'+id+'/'
    print(URL)
    try:
        with urlopen(URL) as res:
            # 本文の抽出
            output1 = ''
            html = res.read().decode('euc_jp', 'ignore')
            soup = BeautifulSoup(html, 'html.parser')
            lineList = soup.select('.articleBody p')
            for line in lineList:
                if len(line.contents) > 0 and type(line.contents[0]) == NavigableString:
                    output1 += line.contents[0].strip()
            if output1 == '': # 記事がない
                return
            output1 += '\n'

            # 要約の抽出
            output0 = ''
            summaryList = soup.select('.summaryList li')
            for summary in summaryList:
                output0 += summary.contents[0].strip()+'\t'
            if output0 == '': # 記事がない
                return

            # 出力
            print(output0+output1)
            with open('./work/output.tsv', mode='a', encoding='utf-8') as f:
                f.writelines(output0+output1)
    except Exception:
        print('Exception')

# IDリストの生成の取得
idList = []
with open('./work/ThreeLineSummaryDataset/data/train.csv', mode='r') as f:
    lines = f.readlines()
    for line in lines:
        id = line.strip().split(',')[3].split('.')[0]
        idList.append(id)

# コンテンツの取得
for i in range(start_index, end_index):
    print('index:', i)
    get_content(idList[i])
