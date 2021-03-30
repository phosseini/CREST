# CREST: A Causal Relation Schema for Text :rocket:

CREST is created to help researchers who work on causal/counterfactual relation extraction/classification, commonsense reasoning, and reading comprehension in natural language to communicate easier and leverage the scattered data resources around this topic. CREST is a user-friendly machine-readable format stored as pandas [DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html).

### `CREST` format
Each relation in a CREST-formatted DataFrame has the following fields/values:
* **`original_id`**: the id of a relation in the original dataset, if such an id exists.
* **`span1`**: a list of strings of the first span/argument of the relation.
* **`span2`**: a list of strings of the second span/argument of the relation
* **`signal`**: a list of strings of signals/markers of the relation in context, if any.
* **`context`**: a text string of the context in which the relation appears.
* **`idx`**: a dictionary in form of `{'span1': [], 'span2': [], 'signal': []}` to store indexes of span1, span2, and signal in context. Each value in the idx dictionary is a list of lists of start and end indexes of spans and the signal. For example, if span1 has multi tokens in context with `start:end` indexes 2:5 and 10:13, respectively, span1's value in `idx` will be `[[2, 5], [10, 13]]`. Lists are sorted based on the start indexes of tokens.
* **`label`**: label of the relation, `0: non-causal`, `1: causal`
* **`direction`**: direction between span1 and span2. `0: span1 => span2`, `1: span1 <= span2`, `-1: not-specified`
* **`source`**: id of the source dataset (`ids` are listed in a table below)
* **`split`**: `0: train`, `1: dev`, `test: 2`. This is the split to which the relation belongs in the original dataset. If there no split specified for a relation in the original dataset, we assign the relation to the `train` split by default.

**Note:** The reason we save a list of strings instead of a single string for span1, span2, and signal is that these text spans may contain multiple non-consecutive sub-spans in context.


### `CREST` conversion
We provide helper methods to convert CREST-formatted data to popular formats and annotation schemes, mainly formats that are used across relation extraction/classification tasks. In the following, there is a list of formats for which we have already developed CREST converter methods:
* `brat`: we have provided helper methods to convert CREST-formatted data frames to brat (brat to CREST converters will be added soon). [brat](https://brat.nlplab.org/) is a popular web-based annotation tool that has been used for a variety of relation extraction NLP tasks. We use brat for two main reasons: 1) better visualization of causal and non-causal relations and their arguments, and 2) modifying relations annotations if needed and adding new annotations to provided context. In the following, there is a sample of a converted version of CREST-formatted relation to brat (example is taken from CaTeRS dataset):
           <p align="center">
           <img src='data/crest_brat_example.png' width='700' height='150' style="vertical-align:middle;margin:100px 50px">
           </p>
* `TACRED`: [TACRED](https://nlp.stanford.edu/projects/tacred/) is a large-scale relation extraction dataset. We convert samples from CREST to TACRED since TACRED-formatted data can be easily used as input to many transformers-based language models.

### Available Data Resources
List of data resources already converted to CREST format:

| Id | Data resource  | Samples | Causal | Non-causal | Availability |
| -- | -------------- | :----------: | :---------: | :-----------: | ------------ |
| 1 | [SemEval 2007 task 4](https://www.aclweb.org/anthology/S07-1003/) | 1,529 | 114 | 1,415 | Public |
| 2 | [SemEval 2010 task 8](https://www.aclweb.org/anthology/S10-1006/) | 10,717 | 1,331 | 9,386 | Public | 
| 3 | [EventCausality](https://cogcomp.seas.upenn.edu/page/resource_view/27) | 485 | 485 | - | Public |
| 4 | [Causal-TimeBank](https://hlt-nlp.fbk.eu/technologies/causal-timebank) | 318 | 318 | - | Not Public | 
| 5 | [EventStoryLine v1.5](https://github.com/tommasoc80/EventStoryLine) | 2,608 | 2,608 | - | Public | 
| 6 | [CaTeRS](https://www.cs.rochester.edu/nlp/rocstories/CaTeRS/) | 2,502 | 308 | 2,194 | Public | 
| 7 | [BECauSE v2.1](https://github.com/duncanka/BECAUSE) | 729 | 554 | 175 | Partially Public| 
| 8 | [Choice of Plausible Alternatives (COPA)](https://www.cs.york.ac.uk/semeval-2012/task7/index.php%3Fid=data.html) | 2,000 | 1,000 | 1,000 | Public |
| 9 | [The Penn Discourse Treebank (PDTB) 3.0](https://catalog.ldc.upenn.edu/LDC2019T05) | 7,991 | 7,991 | - | Not Public |

### How you can contribute:
* Are there any related datasets you don’t see in the list? Let us know or feel free to submit a `Pull Request (PR)`, we actively check the PRs and appreciate it :relaxed:
* Is there a well-known or widely-used machine-readable format you think can be added? We can add the helper methods for conversion or we appreciate PRs.

### How to cite CREST if you found it useful?
For now, please cite our [arXiv paper](https://arxiv.org/abs/2103.13606):
```bibtex
@article{hosseini2021predicting,
  title={Predicting Directionality in Causal Relations in Text},
  author={Hosseini, Pedram and Broniatowski, David A and Diab, Mona},
  journal={arXiv preprint arXiv:2103.13606},
  year={2021}
}
```
