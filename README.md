# EDGE
This is the official implementation of **Post-Training Embedding Enhancement for Long-Tail Recommendation**, (Under Submission to CIKM 2024 Short Paper Track).

## Supplementary Document
The [supplementary document](supplementary.pdf) provides additional details and experimental results to support the main paper.

## Datasets
We use three datasets, **gowalla**, **yelp2018**, and **ml10m**.
Each dataset is split into train/validation/test sets under two test settings, **unbiased (fair)** and **biased (bias)** settings.
The datasets are in the [dataset](dataset) folder.

## Run the Base Model (e.g., LightGCN)
`python main.py --dataset [fair/bias]_[gowalla/yelp2018/ml10m]`

For example, to run **gowalla** on the **ubiased** test setting, run:

`python main.py --dataset fair_gowalla`

The learned embeddings are in [embs](embs) folder.

## Run EDGE
`python edge.py --dataset [fair/bias]_[gowalla/yelp2018/ml10m] --alpha [alpha] --beta [beta] --tau [tau] --lmbda [lambda]`

For example, to run **ml10m** on the **biased** test setting, run:

`python edge.py --dataset bias_ml10m --alpha 0.8 --beta 0.2 --tau 0.2 --lmbda 0.4`

The configuration of each dataset can be found in **run.sh**. You can simply execute `./run.sh`.

## Acknowledgement
This code is implemented based on the [open source code](https://github.com/ml-postech/tten) from the paper **Test Time Embedding Normalization for Popularity Bias Mitigation**.
