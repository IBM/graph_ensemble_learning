# Ensembling Graph Predictions for AMR Parsing

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ensembling-graph-predictions-for-amr-parsing/amr-parsing-on-ldc2017t10)](https://paperswithcode.com/sota/amr-parsing-on-ldc2017t10?p=ensembling-graph-predictions-for-amr-parsing)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ensembling-graph-predictions-for-amr-parsing/amr-parsing-on-ldc2020t02)](https://paperswithcode.com/sota/amr-parsing-on-ldc2020t02?p=ensembling-graph-predictions-for-amr-parsing)



This is the repo for GRAPHENE (Ensembling Graph Predictions for AMR Parsing), a novel approach to ensemble semantic parsing and generation, to be presented at [NeurIPS 2021](https://nips.cc/Conferences/2021/). A preprint of the paper can be found at the [following location on arxiv](https://arxiv.org/abs/2110.09131).


With GRAPHENE you can perform state-of-the-art Text-to-AMR ensemble parsing. If you find it useful please star our github repo and cite our work using the following bib file:

```
@inproceedings{graphene,
      title={Ensembling Graph Predictions for AMR Parsing}, 
      author={Hoang Thanh Lam and Gabriele Picco and Yufang Hou and Young-Suk Lee and Lam M. Nguyen and Dzung T. Phan and Vanessa López and Ramon Fernandez Astudillo},
      year={2021},
      booktitle = {Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual}
}
```

### Abstract

In many machine learning tasks, models are trained to predict structure data such as graphs. For example, in natural language processing, it is very common to parse texts into dependency trees or abstract meaning representation graphs (AMR). On the other hand, ensemble methods combine predictions from multiple models to create a new one that is more robust and accurate than any individual prediction. In the literature, there are many ensembling techniques proposed for classification or regression problems, however, the problem of ensemble graph prediction has not been studied thoroughly. In this work, we formalize this problem as mining the largest subgraph that is the most supported by a collection of graph predictions. As the problem is NP-Hard, we propose an efficient heuristic algorithm to approximate the optimal solution. To validate our approach, we carried out experiments in AMR parsing problems. The experimental results demonstrate that the proposed approach can combine the strength of the state-of-the-art AMR parsers to create new predictions that are more accurate than any individual model in five standard benchmark datasets.
 
## Table of Contents

- [Installation](#installation)
- [Models training](#train-the-models-and-obtain-prediction)
   - [T5 Training](#t5-training)
      - [T5 Scoring](#t5-scoring)
   - [Spring, Cai&Lam and ATP Predictions](#spring,-cai&lam-and-atp-predictions)
- [Run Graphene](#run-graphene)
- [Graphene Evaluation](#graphene-evaluation)
- [Align amr files](#align-amr-files)

## Installation

    pip install -r  requirements.txt

## Train the models and obtain prediction

In order to launch GRAPHENE it is necessary to obtain predictions from a set of models. In the paper, the models we used are:

- [Spring](https://www.researchgate.net/publication/348305083_One_SPRING_to_Rule_Them_Both_Symmetric_AMR_Semantic_Parsing_and_Generation_without_a_Complex_Pipeline)
- [T5](https://arxiv.org/abs/1910.10683)
- [APT](https://arxiv.org/abs/2104.14674)
- [Cai&Lam](https://arxiv.org/abs/2004.05572)

### T5 Training

1.  Obtain the AMR2.0 and AMR3.0 data set: https://catalog.ldc.upenn.edu/LDC2017T10 and https://catalog.ldc.upenn.edu/LDC2020T02

2.  Obtain BIO and LP AMR data set following: https://amr.isi.edu/download.html  

3. Preprocess the data to remove wiki tags. Wiki tags point to reference in Wikipedia, this models perform wikification using Blink as a postprocessing step.
   
   The following command shows how to preprocess AMR 2.0. It can be performed in a similar way on the other datasets.
        
        python -u -m amr_utils.preprocess.preprocess_amr -i LDC2017T10/data/amrs/split \
            -o LDC2017T10/preprocessed_data/

4. Train T5 models and obtaining predictions

        python -u -m amr_parsing.t5.cli.train --train "./LDC2017T10/preprocessed_data/train.txt.features.nowiki" \
            --validation ./LDC2017T10/preprocessed_data/dev.txt.features.nowiki \
            --report_test ./LDC2017T10/preprocessed_data/test.txt.features.nowiki \
            --max_source_length 512 --max_target_length 512 --batch 8 -e 30 -m t5-large \
            --model_type t5 --output ./t5_amr/ --data_type "amrdata" \
            --task_type "text2amr" --val_from_epoch 10

   * Multiple T5 models can be trained for running the graph ensemble algorithm, using different random seed with the --random_seed option
   

#### T5 obtaining predictions from a checkpoint
1. Scoring

         python -u -m amr_parsing.cli.parser --test LDC2017T10/preprocessed_data/test.txt.features.nowiki \
             --max_source_length 512 --max_target_length 512 --batch 4 -m t5-large --model_type t5  \
             --output LDC2017T10/preprocessed_data/t5_amr_prediction.txt --data_type "amrdata" --task_type "text2amr" \
             --checkpoint t5_amr/multitask.model
   
2. Wikification

    To reproduce our results, you will also need need to run the [BLINK](https://github.com/facebookresearch/BLINK) 
    entity linking system on the prediction file. To do so, you will need to install BLINK, and download their models:
    ```shell script
    git clone https://github.com/facebookresearch/BLINK.git
    cd BLINK
    pip install -r requirements.txt
    pip install -e .
    sh download_blink_models.sh
    cd models
    wget http://dl.fbaipublicfiles.com/BLINK//faiss_flat_index.pkl
    cd ../..
    ```
    Then, you will be able to launch the `run_blink_wiki_adder.py` script:
    ```shell
    python -u -m amr_utils.blinkify.run_blink_wiki_adder.py \
    -i LDC2017T10/preprocessed_data/t5_amr_prediction.txt \ 
    -o LDC2017T10/preprocessed_data/ \
    --blink-models-dir ../BLINK/models/ 

The output file with wikifications will be written to the output folder.

### Spring, Cai&Lam and ATP Predictions

For Spring, ATP and Cai&Lam we used the pretrained available checkpoints.

- For Spring refer to: https://github.com/SapienzaNLP/spring
- For Cai&Lam refer to: https://github.com/jcyk/AMR-gs
- For ATP refer to: https://github.com/IBM/transition-amr-parser

## Run Graphene

    python -u -m ensemble.graphene --gold gold_amr.txt --data "prediction_model_1 prediction_model_2 ...." 
 
 The results will be written to the default output file graphene_smatch.txt in the same folder where the command is run. To choose which algorithms, we can specify the algorithm flag with the following options:
 
 - vote: the first graph in the input is chosen as the pivot graph, and that graph is modified and chosen as the final prediction.
 - graphene: every input graph is chosen as a pivot graph once and the best among the modified pivot graphs is chosen as the final prediction based on average support.  
 - graphene_smatch (default): similar to graphene except in the last step, the best modified pivot graph was chosen based on average Smatch similar to Barzdins et al. rather than based on support. 
 
## Graphene Evaluation

To compute comparable Smatch scores you will also need to use the scripts available at https://github.com/mdtux89/amr-evaluation. It is important to notice that the results collected using this script is about 0.3 points worse than the results using Smatch 1.0.4. Following https://github.com/SapienzaNLP/spring, the results reported in our paper are based on https://github.com/mdtux89/amr-evaluation, instead of Smatch 1.0.4.

### Align amr files

The script provided in https://github.com/mdtux89/amr-evaluation require the files to be aligned (AMR provided in the prediction and gold files must be in the same order). We provide an utility for aligning two amrs file based on the ::id tags:

      python -u -m ensemble.align -g gold_amrs.txt -p predictions_amrs.txt
