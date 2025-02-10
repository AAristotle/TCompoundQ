# **TCompoundQ: Translation, Rotation, and Scaling in Quaternion Vector Space for Temporal Knowledge Graph Completion**  

This repository contains the source code for our **DASFAA 2025** paper:  
**"TCompoundQ: Translation, Rotation, and Scaling in Quaternion Vector Space for Temporal Knowledge Graph Completion."**  

The complete code will be released upon the official publication of the paper.  


## **Environment Setup**  

To ensure reproducibility, install the required dependencies using:  
```
pip install -r requirements.txt
tqdm==4.59.0
numpy==1.20.1
scikit-learn==0.24.1
scipy==1.6.2
torch==1.9.0
```

## **Datasets**
Once the datasets are downloaded, go to the tkbc/ folder and add them to the package data folder by running :
```
python process_icews.py
python process_timegran.py --tr 100 --dataset yago11k
python process_timegran.py --tr 1 --dataset wikidata12k
```


## **Reproducing results of TCompoundQ**

```
python learner.py --dataset ICEWS14 --model TCompoundQ --rank 6000 --emb_reg 0.003 --time_reg 0.003 

python learner.py --dataset ICEWS05-15 --model TCompoundQ --rank 6000 --emb_reg 0.0000006 --time_reg 0.006

python learner.py --dataset GDELT --model TCompoundQ --rank 6000  --emb_reg 0.0001 --time_reg 0.05

python learner.py --dataset yago11k --model TCompoundQ --rank 6000 --emb_reg 0.05 --time_reg 0.005

python learner.py --dataset wikidata12k --model TCompoundQ --rank 6000 --emb_reg 0.06 --time_reg 0.005

```

## **Acknowledgement**
This project is inspired by and builds upon previous works, including TNTComplEx, TeLM, and TeAST. We sincerely appreciate their contributions to the field.

## **References**
https://github.com/facebookresearch/tkbc.
Xu, Chengjin, et al. "Temporal knowledge graph completion using a linear temporal regularizer and multivector embeddings." Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. 2021.
Ying, Rui, et al. "Simple but Effective Compound Geometric Operations for Temporal Knowledge Graph Completion." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.
