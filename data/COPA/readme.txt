COPA-resources readme.txt
-------------------------

This archive contains materials related to the Choice Of Plausible Alternatives (COPA) evaluation of open-domain commonsense causal reasoning. 

datasets/copa-dev.xml : 500 questions of the development set
datasets/copa-test.xml : 500 questions of the test set
datasets/copa-all.xml : 1000 questions of both the development and test sets
datasets/copa.dtd : The format of the XML question files
results/gold-*.txt : Correct answers for each set of questions
results/baselineFirst-*.txt : Choices where the first alternative is always selected
results/PMIGutenbergW5-*.txt : Choices made by the best-performing baseline system of Roemmele et al, 2011.
copa-eval.jar : A java package for computing statistical significance of differences in answer sets
copa-eval.sh : A simple shell script for using the java package

For more information, please check out the following webpage:
http://people.ict.usc.edu/~gordon/copa.html

Also, please see the following publication:
Roemmele, M., Bejan, C., and Gordon, A. (2011) Choice of Plausible Alternatives: An Evaluation of Commonsense Causal Reasoning. AAAI Spring Symposium on Logical Formalizations of Commonsense Reasoning, Stanford University, March 21-23, 2011.


--
Version 1. March 2011