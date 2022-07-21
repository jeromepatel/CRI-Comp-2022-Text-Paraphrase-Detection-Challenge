# CRI-Comp-2022-Text-Paraphrase-Detection-Challenge - First Rank winner solution
 Solution and experiments to the challenge CRI competition 2022 - text paraphrase detection
## This repo contains raw code for winning solution of [CRI-COMP-2022 Text Paraphrase challenge](https://sites.google.com/g.syr.edu/cri-comp-2022/text-paraphrase-detection-challenge). 
### I'll add cleaned code in a separate repo with training instructions, however this repo contains code of all of my experiments, including failed ones. 
 * This repo includes the baseline approach using sentenceTransformer module.
 * Using paraphrase_mining function which calculates sentence embeddings first and uses cosine sim distance to get paraphrase for query sentence.
 * Please refer file `bertbaseline.py` for more details.
 * Update: check the jupyter notebook which can directly be ran in colab, the final solution contains two stages, first stage uses fine-tuned bi-encoder embeddings and second stage uses pretrained cross-encoder pairs with custom postprocessing to get the results. 
 * Final solution scored 0.78 F1-score on private leaderboard (test set) with rank 1. 
 
