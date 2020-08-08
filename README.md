# CREST: A Causal Relation Schema for Text

CREST is created to help researchers who work on causal/counterfactual relation extraction, commonsense reasoning, and reading comprehension in natural language to communicate easier and leverage all the created data resources around this topic. CREST is a user friendly machine-readable format stored as pandas [DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) to unify all the scattered datasets of causal and non-causal samples.

### `CREST` convertion
We convert CREST-formatted data to popular formats and annotation schemes, mainly formats that are used across relation extraction tasks. In the following, there is a list of formats for which we have already developed CREST converter methods:
* `brat`: we have provided helper methods to convert CREST-formatted data frames to brat and also convert the brat annotation version back to CREST. [brat](https://brat.nlplab.org/) is a popular web-based annotation tool that has been used for a variety of relation extraction NLP tasks. We use brat for two main reasons: 1) better visualization of causal and non-causal relations and their arguments, and 2) modifying relations annotations if needed and adding new annotations to provided context. In the following, there is a sample of a converted version of CREST-formatted relation to brat (example is taken from CaTeRS dataset):
           <p align="center">
           <img src='data/crest_brat_example.png' width='700' height='150' style="vertical-align:middle;margin:100px 50px">
           </p>
* `TACRED`: [TACRED](https://nlp.stanford.edu/projects/tacred/) is a large-scale relation extraction dataset. We convert samples from CREST to TACRED since TACRED-formatted data can be easily used as input to many transformers-based language models. These models include but not limited to [BERT](https://github.com/google-research/bert) and [SpanBERT](https://github.com/facebookresearch/SpanBERT) that we have already fine-tuned them for classifying causal and non-causal relation classification.

### Available Data Resources
List of data resources already converted to CREST format:

| Id | Data resource  | samples | causal | non-causal | availability |
| -- | -------------- | :----------: | :---------: | :-------------: | ----------------- |
| 1 | SemEval 2007 task 4 | 220 | 114 | 106 | Public |
| 2 | SemEval 2010 task 8 | 10,717 | 1,331 | 9,386 | Public |
| 3 | EventCausality | 485 | 485 | - | Public |
| 4 | Causal-TimeBank | 318 | 318 | - | Not Public|
| 5 | EventStoryLine v1.5 | 2608 | 2608 | - | Public |
| 6 | CaTeRS | 2502 | 308 | 2194 | Public |
| 7 | BECauSE v2.1 | 729 | 554 | 175 | Partially Public|
| 8 | Choice of Plausible Alternatives (COPA) | 2000 | 1000 | 1000 | Public |
