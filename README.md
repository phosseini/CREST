# CREST: A Causal Relation Schema for Text :rocket:

CREST is a machine-readable format/schema that is created to help researchers who work on causal/counterfactual relation extraction and commonsense causal reasoning, to use and leverage the scattered data resources around these topics more easily. CREST-formatted data are stored as pandas [DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html).

### How to convert dataset(s) to CREST:
* Clone this repository and go to the `/CREST` directory.
* Install the requirements: `pip install -r requirements.txt`
* Download spaCy's model: `python -m spacy download en_core_web_sm`
* Run the [`/crest/convert.py`](https://github.com/phosseini/CREST/blob/master/crest/convert.py):
     * `python convert.py -i`: printing the full list of currently supported datasets
     * `python convert.py [DATASET_ID_0] ... [DATASET_ID_n] [OUTPUT_FILE_NAME]`
          * `DATASET_ID_*`: id of a dataset.
          * `OUTPUT_FILE_NAME`: name of the output file that should be in `.xlsx` format
 * **Examples:**
     * Converting datasets `1` and `2`: `python convert.py 1 2 output.xlsx`
     * Converting dataset `5`: `python convert.py 5 output.xlsx`

> The excel file of all converted datasets: [`crest_v2.xlsx`](https://github.com/phosseini/CREST/blob/master/data/crest_v2.xlsx)
> * PDTB is not available in this file due to copyright. However, you can still use CREST to convert this dataset if you have access to PDTB.

### `CREST` format
Each relation in a CREST-formatted DataFrame has the following fields/values:
* **`original_id`**: the id of a relation in the original dataset, if such an id exists.
* **`span1`**: a list of strings of the first span/argument of the relation.
* **`span2`**: a list of strings of the second span/argument of the relation
* **`signal`**: a list of strings of signals/markers of the relation in context, if any.
* **`context`**: a text string of the context in which the relation appears.
* **`idx`**: indices of `span1`, `span2`, and `signal` tokens/spans in context stored in 3 lines, each line in the form of `span_type start_1:end_1 ... start_n:end_n`. For example, if `span1` has multiple tokens/spans with `start:end` indices `2:5` and `10:13`, respectively, `span1`'s line value in `idx` is `span1 2:5 10:13`. Indices are sorted based on the start indexes of tokens/spans.
* **`label`**: label of the relation, `0: non-causal`, `1: causal`
* **`direction`**: direction between span1 and span2. `0: span1 => span2`, `1: span1 <= span2`, `-1: not-specified`
* **`source`**: id of the source dataset (`ids` are listed in a table below)
* **`split`**: `0: train`, `1: dev`, `2: test`. This is the split to which the relation belongs in the original dataset. If there is no split specified for a relation in the original dataset, we assign the relation to the `train` split by default.

**Note:** The reason we save a list of strings instead of a single string for span1, span2, and signal is that these text spans may contain multiple non-consecutive sub-spans in context.

### Available Data Resources
List of data resources already converted to CREST format:

| Id | Data resource  | Samples | Causal | Non-causal | Document | Year |
| -- | -------------- | :----------: | :---------: | :-----------: | :-----------: | :--: |
| 1 | [SemEval 2007 Task 4](https://www.aclweb.org/anthology/S07-1003/) | 1,529 | 114 | 1,415 | [Paper](https://aclanthology.org/S07-1003/) | 2007 |
| 2 | [SemEval 2010 Task 8](https://www.aclweb.org/anthology/S10-1006/) | 10,717 | 1,331 | 9,386 | [Paper](https://aclanthology.org/S10-1006/) | 2010 |
| 3 | [EventCausality](https://cogcomp.seas.upenn.edu/page/resource_view/27) | 583 | 583 | - | [Paper](https://aclanthology.org/D11-1027/) | 2011 |
| 4 | [Causal-TimeBank](https://github.com/paramitamirza/Causal-TimeBank) | 318 | 318 | - | [Paper](https://aclanthology.org/W14-0702/) | 2014 |
| 5 | [EventStoryLine v1.5](https://github.com/tommasoc80/EventStoryLine) | 2,608 | 2,608 | - | [Paper](https://aclanthology.org/W17-2711/) | 2016 |
| 6 | [CaTeRS](https://www.cs.rochester.edu/nlp/rocstories/CaTeRS/) | 2,502 | 308 | 2,194 | [Paper](https://www.usna.edu/Users/cs/nchamber/pubs/naacl2016-caters.pdf) | 2016 |
| 7 | [BECauSE v2.1](https://github.com/duncanka/BECAUSE) <sup>:warning:</sup> | 729 | 554 | 175 | [Paper](https://aclanthology.org/W17-0812/) | 2017 |
| 8 | [Choice of Plausible Alternatives (COPA)](https://people.ict.usc.edu/~gordon/copa.html) | 2,000 | 1,000 | 1,000 | [Paper](https://people.ict.usc.edu/~gordon/publications/AAAI-SPRING11A.PDF) | 2011 |
| 9 | [The Penn Discourse Treebank (PDTB) 3.0](https://catalog.ldc.upenn.edu/LDC2019T05) <sup>:warning:</sup> | 7,991 | 7,991 | - | [Manual](https://catalog.ldc.upenn.edu/docs/LDC2019T05/PDTB3-Annotation-Manual.pdf) | 2019 |
| 10 | [BioCause Corpus](http://www.nactem.ac.uk/biocause/) | 844 | 844 | - | [Paper](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-14-2) | 2013 |
| 11 | [Temporal and Causal Reasoning (TCR)](https://cogcomp.seas.upenn.edu/page/resource_view/118) | 172 | 172 | - | [Paper](https://aclanthology.org/P18-1212/) | 2018 |
| 12 | [Benchmark Corpus for Adverse Drug Effects](https://sites.google.com/site/adecorpus/) | 5,671 | 5,671 | - | [Paper](https://www.sciencedirect.com/science/article/pii/S1532046412000615) | 2012 |
| 13 | [SemEval 2020 Task 5](https://github.com/arielsho/SemEval-2020-Task-5) <sup>:atom:</sup>| 5,501 | 5,501 | - | [Paper](https://aclanthology.org/2020.semeval-1.40/) | 2020 |

:warning:&nbsp;The data is either not publicly available or partially available. You can still use CREST for conversion if you have full access to this dataset.

:atom:&nbsp;&nbsp;Counterfactual Relations

### `CREST` conversion
We provide helper methods to convert CREST-formatted data to popular formats and annotation schemes, mainly formats that are used across relation extraction/classification tasks. In the following, there is a list of formats for which we have already developed CREST converter methods:
* `brat`: we have provided helper methods for two-way conversion of CREST data frames to brat (see example [here](https://github.com/phosseini/CREST/blob/master/notebooks/crest_brat.ipynb)). [brat](https://brat.nlplab.org/) is a popular web-based annotation tool that has been used for a variety of relation extraction NLP tasks. We use brat for two main reasons: 1) better visualization of causal and non-causal relations and their arguments, and 2) modifying annotations if needed and adding new annotations to provided context. In the following, there is a sample of a converted version of CREST-formatted relation to brat (example is taken from CaTeRS dataset):
           <p align="center">
           <img src='data/crest_brat_example.png' width='700' height='150' style="vertical-align:middle;margin:100px 50px">
           </p>
* `TACRED`: [TACRED](https://nlp.stanford.edu/projects/tacred/) is a large-scale relation extraction dataset. We convert samples from CREST to TACRED since TACRED-formatted data can be easily used as input to many transformers-based language models (e.g. for Relation Classification/Extraction). You can find an example of converting CREST-formatted data to TACRED in [this notebook](https://github.com/phosseini/CREST/blob/master/notebooks/crest_tacred.ipynb).

### How you can contribute:
* Are there any related datasets you don’t see in the list? Let us know or feel free to submit a `Pull Request (PR)`, we actively check the PRs and appreciate it :relaxed:
* Is there a well-known or widely-used machine-readable format you think can be added? We can add the helper methods for conversion or we appreciate PRs.

### How to cite CREST?
For now, please cite our [arXiv paper](https://arxiv.org/abs/2103.13606):
```bibtex
@article{hosseini2021predicting,
  title={Predicting Directionality in Causal Relations in Text},
  author={Hosseini, Pedram and Broniatowski, David A and Diab, Mona},
  journal={arXiv preprint arXiv:2103.13606},
  year={2021}
}
```
