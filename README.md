# SciSumLongFormer

guidelines for summarizing: https://huggingface.co/course/chapter7/5?fw=pt

seq2seq model to fine: https://huggingface.co/Dagar/t5-small-science-papers , https://huggingface.co/L-macc/autotrain-Biomedical_sc_summ-1217846148 

dataset: https://huggingface.co/datasets/scientific_papers OR https://huggingface.co/datasets/tomasg25/scientific_lay_summarisation

transformer into longformer: https://github.com/allenai/longformer/issues/64

--------

code example 1 => (baseline for architecture 2): https://github.com/christianversloot/machine-learning-articles/blob/main/transformers-for-long-text-code-examples-with-longformer.md

--------

RESEARCH IDEA AND WHERE TO LOOK INTO:
1) Architecture 1: possible clustering of abstract and identify cluster, so split paper and federate learning
2) Architecture 2: transform the seq2seq model fined tuned on scientific paper into a longformer, compare it with the performance of a normal longformer

