# RCENR_code
This repository is for the explainable news recommendation which is accepted in SIGIR 2023:

RCENR: A Reinforced and Contrastive Heterogeneous NetworkReasoning Model for Explainable News Recommendation.

In this paper, we research the news recommendation with heterogeneous news network reasoning to improve recommendation accuracy and explainability.

@inproceedings{10.1145/3539618.3591753,
author = {Jiang, Hao and Li, Chuanzhen and Cai, Juanjuan and Wang, Jingling},
title = {RCENR: A Reinforced and Contrastive Heterogeneous Network Reasoning Model for Explainable News Recommendation},
year = {2023},
isbn = {9781450394086},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3539618.3591753},
doi = {10.1145/3539618.3591753},
abstract = {Existing news recommendation methods suffer from sparse and weak interaction data, leading to reduced effectiveness and explainability. Knowledge reasoning, which explores inferential trajectories in the knowledge graph, can alleviate data sparsity and provide explicitly recommended explanations. However, brute-force pre-processing approaches used in conventional methods are not suitable for fast-changing news recommendation. Therefore, we propose an explainable news recommendation model: the Reinforced and Contrastive Heterogeneous Network Reasoning Model for Explainable News Recommendation (RCENR), consisting of NHN-R2 and MR\&CO frameworks. The NHN-R2 framework generates user/news subgraphs to enhance recommendation and extend the dimensions and diversity of reasoning. The MR\&CO framework incorporates contrastive learning with a reinforcement-based strategy for self-supervised and efficient model training. Experiments on the MIND dataset show that RCENR is able to improve recommendation accuracy and provide diverse and credible explanations.},
booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {1710–1720},
numpages = {11},
keywords = {markov decision process, contrastive learning, knowledge reasoning, news recommendation, explainable recommendation},
location = {Taipei, Taiwan},
series = {SIGIR '23}
}
