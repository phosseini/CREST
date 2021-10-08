# BECAUSE v2.1 Data Release

This directory contains the annotated data from version 2.1 of the **B**ank of **E**ffects and **Cau**ses **S**tated **E**xplicitly (BECAUSE).<sup>[1](#footnote1)</sup> This is the exhaustively annotated dataset reported on in [Dunietz et al., 2017b](https://www.cs.cmu.edu/~jdunietz/publications/because-v2.pdf) -- an expanded and improved version of the dataset reported on in [Dunietz et al., 2017a](https://www.cs.cmu.edu/~jdunietz/publications/causeway-system.pdf). Version 2.1 includes a few small corrections to the annotations. (You can find the data from the 1.0 and 2.0 releases in their [own](https://github.com/duncanka/BECauSE/tree/1.0) [branches](https://github.com/duncanka/BECauSE/tree/2.0).)

The corpus is an attempt to capture the enormous variety of constructions used to express cause and effect. Inspired by the principles of [Construction Grammar](https://en.wikipedia.org/wiki/Construction_grammar), we annotate any conventional pattern that expresses causation, however complex. BECAUSE thus includes many constructions that are not annotatable in most schemes, and is more comprehensive than previous efforts to capture causal language. This version also includes annotations for seven types of semantic relations that frequently overlap with causality and are sometimes used to express it. For details, see the aforementioned papers.

The list of causal constructions is available in the [constructicon](https://docs.google.com/spreadsheets/d/1oGmrdLIruo32okPcFSCERupOuepiPwSD96H_WVTq10E/edit) used by annotators for this release.

All annotations are in `.ann` files formatted for [brat](http://brat.nlplab.org/), and we have included the `annotation.conf` and `visual.conf` files for brat in this directory. There are four data subdirectories, each containing data from a different source:

* **CongressionalHearings**: Three partial documents from the [2014 NLP Unshared Task in PoliInformatics](https://sites.google.com/site/unsharedtask2014/). These documents are freely available, but for ease of processing, some header information was stripped from the text files. We also annotated only portions of these files, not complete transcripts. To allow for others to use our annotation offsets, then, we have included the preprocessed text files alongside the annotations.

* **NYT**: 59 randomly selected documents from the year 2007 of the Washington section of the [New York Times Annotated Corpus](https://catalog.ldc.upenn.edu/ldc2008t19) (Sandhaus, 2008). Each `.ann` file shares its name with the corresponding raw NYT article. An LDC subscription is required to obtain the raw files.

  To turn the raw XML into the plain text files that the annotation offsets correspond to, you will need to run [`extract_nyt_txt.py`](scripts/extract_nyt_txt.py) on a system with access to `sed`. The script depends on NLTK.

* **PTB**: 47 documents randomly selected from sections 2-23 of the Penn Treebank (Marcus et al., 1994). We excluded WSJ documents that were either earnings reports or corporate leadership/structure announcements, as both tended to be merely short lists of names/numbers. Again, we provide offset annotations named to match the raw PTB files, but the raw files require an LDC subscription.

  To place the original plain-text of the PTB files alongside their corresponding annotations, run the following script (assuming BECAUSE is saved in `$BECAUSE_DIR` and the PTB CD has been extracted into `PTB_DIR`):
  ```bash
  for ANN_FILE in $BECAUSE_DIR/PTB/*.ann; do
      BASE_FILE=$(basename $ANN_FILE)
      DIGITS=$(echo $BASE_FILE | cut -d'_' -f2)
      cp $PTB_DIR/raw/${DIGITS:0:2}/${BASE_FILE%.*}.txt $BECAUSE_DIR/PTB/
  done
  ```

* **MASC**: 10 newspaper documents (Wall Street Journal and New York Times articles, totalling 547 sentences) and 2 journal documents (82 sentences) from the Manually Annotated Sub-Corpus (MASC; Ide et al., 2010).

The first three sets of documents are the same dataset that was annotated for BECAUSE 1.0.

<sub><a name="footnote1">[1]</a>: Prior to May 2017, the corpus was referred to with a different capitalization scheme (BECauSE). The official name for the corpus is now BECAUSE, but either format is fine for citation.</sub>

#### Citations

<sub>Dunietz, Jesse, Lori Levin, and Jaime Carbonell. The BECauSE Corpus 2.0: Annotating Causality and Overlapping Relations. *Proceedings of LAW XI – The 11th Linguistic Annotation Workshop* (2017b): Association for Computational Linguistics, Valencia, Spain.</sub>

<sub>Dunietz, Jesse, Lori Levin, and Jaime Carbonell. Automatically Tagging Constructions of Causation and Their Slot-Fillers. *Transactions of the Association for Computational Linguistics* (2017a): in press.</sub>

<sub>Dunietz, Jesse, Lori Levin, and Jaime Carbonell. Annotating Causal Language Using Corpus Lexicography of Constructions. *Proceedings of LAW IX – The 9th Linguistic Annotation Workshop* (2015): 188-196. Association for Computational Linguistics, Denver, USA.</sub>

<sub>Sandhaus, Evan. 2008. The New York Times annotated corpus. Linguistic Data Consortium, Philadelphia, USA.</sub>

<sub>Nancy Ide, Christiane Fellbaum, Collin Baker, and Rebecca Passonneau. The Manually Annotated Sub-Corpus: A community resource for and by the people. *Proceedings of the 48th Annual Meeting of the Association for Computational Linguistics* (2010): 68-73. Association for Computational Linguistics, Uppsala, Sweden.</sub>

<sub>Marcus, Mitchell, et al. 1994. The Penn Treebank: Annotating predicate argument structure. In *Proceedings of the Workshop on Human Language Technology*, HLT '94, pages 114-119. Association for Computational Linguistics, Plainsboro, USA.</sub>
