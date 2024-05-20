# CdSC
Pytorch codes of 'Cross-Difference Semantic Consistency Network for Semantic Change Detection' [[paper]](https://ieeexplore.ieee.org/document/10494733)


**Data preparation:**
1. Split the SCD data into training, validation and testing (if available) set and organize them as follows:

>YOUR_DATA_DIR
>  - Train
>    - im1
>    - im2
>    - label1
>    - label2
>  - Val
>    - im1
>    - im2
>    - label1
>    - label2
>  - Test
>    - im1
>    - im2
>    - label1
>    - label2
    
2. Find *-datasets -RS_ST.py*, set the data root in *Line 22* as *YOUR_DATA_DIR*

3. The pretrained weights and JL1 dataset can be accessed at “Link: https://pan.baidu.com/s/1R0OZz3y2kIWCRoU_H1bQMg  
  Extract code: nwpu”.

**Reference**

If you find our work useful or interesting, please consider to cite:
> @ARTICLE{10494733,  
  author={Wang, Qi and Jing, Wei and Chi, Kaichen and Yuan, Yuan},  
  journal={IEEE Transactions on Geoscience and Remote Sensing},  
  title={Cross-Difference Semantic Consistency Network for Semantic Change Detection},  
  year={2024},  
  volume={62},  
  number={},    
  pages={1-12},  
  keywords={Semantics;Feature extraction;Task analysis;Self-supervised learning;Data models;Data mining;Solid modeling;Cross-difference;deep learning;remote sensing image;semantic change detection (SCD);semantic consistency},  
  doi={10.1109/TGRS.2024.3386334}}
