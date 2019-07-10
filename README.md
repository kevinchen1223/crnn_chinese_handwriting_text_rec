
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

In Ubuntu, run `unalz HWDB1.1trn_gnt.alz` and ``unalz HWDB1.1tst_gnt.alz``

run `data_generator/preProcessing.py` to extract all 3755 Chinese characters and generate ``charDict.npy`` which is the dictionary mapping folder name to Chinese Characters

------

 To generate texts in Chinese, Randomly pick chars from dataset

run ``data_generator/chinese_text.py``

``data_generator/charDict.npy`` is the char dictionary

------

Here are some samples.

in ``test_width.txt`` 

00001.jpg 符疆葛去卑狂擅改汐堂苯谎粥紫鸣

![](https://github.com/NormXU/crnn_chinese_handwriting_text_rec/blob/master/to_lmdb/test_width/00001.jpg)

**Attention !** To generate lmdb file for crnn training, chinese characters should be stored in UTF-8 format. Try to use NodePad++ to transform ANSI format to UTF-8 format



In ``alphabets.py``, there are two chinese character dictionaries. Switch between each other if you found **IndexError**

For Training, 
Firstly, ``run tolmdb_py3.py`` to compress all data and labels in lmdb format, 
then
```python
python crnn_main.py
```



## Results

The Training Set has 200, 000 Chinese text images. The regression is really slow. It toke around 50 hours on RTX2080Ti

Training Loss:

![](https://github.com/NormXU/crnn_chinese_handwriting_text_rec/blob/master/test_images/loss.png)

#### Testing Accuracy:

#### Example 1

![](https://github.com/NormXU/crnn_chinese_handwriting_text_rec/blob/master/test_images/test1.jpg)

![](https://github.com/NormXU/crnn_chinese_handwriting_text_rec/blob/master/test_images/1.JPG)

### Example 2

![](https://github.com/NormXU/crnn_chinese_handwriting_text_rec/blob/master/test_images/test2.jpg)

![](https://github.com/NormXU/crnn_chinese_handwriting_text_rec/blob/master/test_images/2.JPG)

### Example 3

In ``TestAll.py`` you can simply test the whole dataset with **Batch_Test_Flag = True**

Test the Randomly Generated Dataset, the accuracy is around $$90\%$$

![](https://github.com/NormXU/crnn_chinese_handwriting_text_rec/blob/master/test_images/Accuracy.JPG)
