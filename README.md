# crfasrnn-training

crfasrnnで学習するためのサポートツールです。

## 依存関係(dependencies)

- CRFasRNN https://github.com/torrvision/crfasrnn
- 学習済みモデル
```
#!bash
wget https://s3-ap-northeast-1.amazonaws.com/recognizetrainimages/TVG_CRFRNN_COCO_VOC.caffemodel

```


## 教師データ
- PASCAL VOC 2012形式 (02.05.2016現在)

### PASCAL VOC 2012形式
- VOCdevkit/VOC2012/SegmentationClass:

セグメンテーションされたpng画像が格納されている。例えば人はピンクに領域が塗られているような感じ。

- VOCdevkit/VOC2012/JPEGImages:

オリジナルのJPEG画像  
上記のSegmentationClassと拡張子以外のファイル名が対応している

- 上記のデータセットにアクセスするときのちょっとした工夫

シンボリックを貼っておくと良い


```
#!bash

ln -s ${DATASETS}/VOCdevkit/VOC2012/SegmentationClass labels
ln -s ${DATASETS}/VOCdevkit/VOC2012/JPEGImages images

```

## 分類したいクラスの特定
PASCAL VOCの中から分類したいクラスを特定する
例では3クラスを用いる

segmentationのpngファイルから必要なクラスの画像だけリストする。

まずは全てのSegmentationラベルのリストをつくる


```
#!bash

create_train_txt.sh  <labels image directory path>

```

PASCAL VOC 2012 形式のデータセットだと、真の領域はRGB画像で定義される。しかし、別のデータセットや既に前処理がほどこされた領域を使うことを決めたなら、  
ラベルのインデックスを表すグレースケールの画像でも動作させることができる。

なぜならば、学習のための教師データ生成がいくつかのパートに分かれており、各の画像の画素3チャンネルに2回アクセスする。  
前処理をほどこしていない真の領域で作業する場合は、2回変換しなければならないだろう。  

残念ながらこの変換はけっこう時間がかかる、そのため以下のコマンドを最初に実行することをおすすめする。これは強制ではない。


```
#!bash

convert_labels.sh <labels image directory path> <labels image list path> <output converted image directory path>

```


convert_labels.shでは、RGB情報をグレースケールに変換する処理を行う。  

変換規則はutil.pyで定義され、convert_from_color_segmentation関数で実行できる  
util自体が設定ファイルとgetterを兼ねている。  

興味があるクラス（ここでは分類したいクラス）はfilter_images.pyに記述する。  
このスクリプトは分類したいクラスに対応したいくつかのテキストファイル(求めるクラスを含む画像のリスト)を生成する。  
これらのファイルはtrain.txtと同じ構造をしている。  
実験で様々なクラスを使うので、それぞれのクラスに対してファイルをつくるのが良い。  
  
画像のラベルは興味のあるクラス以外も含まれていることに気づくだろう、するともっとも小さいidのクラスが割り当てられる。  
これらの行動は同じクラスのラベルばかりデータセットに含まれる可能性をもつ。しかしながら、背景クラスとしてはカウントされない。  


```
#!bash

filter_images.sh <labels image directory path> <labels image list path>

```


## LMDBの作成

オリジナルのcrf-rnnは500x500pxの画像を学習に使っていたしここでもそうする。しかし、これといった事情がなければ、  
異なる次元数を選択しても構わない。data2lmdb.pyで変更することができる。現状、500pxより大きくしたことはない。  

以下のコマンドでtrainとtestに使うlmdbを作成できる  


```
#!bash

python data2lmdb.py converted_labels

```

4つのlmdbができる。train、testとそれぞれのオリジナル画像とセグメンテーション画像である。


## 学習
学習済みモデルをファインチューニングする。
オリジナルのCaffeが同居している場合に対応するため、以下の環境変数を設定すること。


```
#!bash

CRF_AS_RNN_PATH=path/to/オリジナルのCRFasRNN実装

```

以下のスクリプトで学習を開始する。


```
#!bash

python solve.py
```
