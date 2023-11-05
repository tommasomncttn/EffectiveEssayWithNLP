# Neural Augmentation for Text Classification

Kaggle Competition: https://www.kaggle.com/competitions/nlp-getting-started

Comparative analysis of the performance of Bert, DistilBert, ULMFiT, and logistic regression (baseline). Models relies on different library (pytorch, hugginface, fastai and sklearn)

----------------------------------------------------------------------------------------------------

## AUGMENTER

To tackle the classification task, a brand new augmentation pipeline was implemented. Initially, a PEGASUS sequence-to-sequence model, fine-tuned on a paraphrasing task, is deployed to get n paraphrases of the input text. By deploying a pre-trained and fine-tuned large  language model to rephrase the input sentence we can guarantee the syntactic and semantic consistency of the generated data, differently from augmentation methods on the word level. The subsequent step involves ranking the n-generated sentences based on their similarity to the original sentence, utilizing cosine similarity. Cosine similarity is a method for determining the similarity between two vectors in a high-dimensional space. It is calculated as the cosine of the angle between the two vectors. To accomplish this, we utilized the Spacy Library and its
“en_core_web_sm” pre-trained model to get embedded sentences and compute the cosine similarity between sentences.
<p align="center">
  <img width="600" src="https://user-images.githubusercontent.com/91601166/230159690-04648428-fe80-4f4a-acff-512070ebe854.png" alt="Word Cloud">
</p>

