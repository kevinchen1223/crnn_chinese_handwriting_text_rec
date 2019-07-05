
# CRNN_Chinese_HandWriting_Text_Recognition



This project is modified from https://github.com/Sierkinhane/crnn_chinese_characters_rec

## Dependence

- [warp_ctc_pytorch](https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings/pytorch_binding)
- lmdb



## Data Generator

The Training dataset is from, use following commands to download

```shell
wget http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1trn_gnt.zip
wget http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1tst_gnt.zip
```



After Extract the files, you will find another compressed files in alz format. 

In Ubuntu, run `unalz HWDB1.1trn_gnt.alz`

run `data_generator/preProcessing.py` to extract all 3755 Chinese characters

 To generate texts in Chinese, Randomly pick chars from dataset

run ``data_generator/chinese_text.py``

Here are some samples.

in ``testUT.txt`` 

00000.jpg 棒摄部籍鼓芥与敬红竣片

![](https://github.com/NormXU/crnn_chinese_handwriting_text_rec/blob/master/to_lmdb/test_width/00000.jpg)

Attention, to generate lmdb file for crnn training, chinese characters should be stored in UTF-8 format. Try to use NodePad++ to transform ANSI format to UTF-8 format



In ``alphabets.py``, there are two chinese character dictionaries. Switch between them if you found **IndexError**

For Training, 

```python
python crnn_main.py
```



## Results

Training Loss:



Testing Accuracy:



