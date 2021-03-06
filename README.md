# IIT(BHU)-IIITH system for CoNLL–SIGMORPHON 2018 Shared Task

This repository contains code of IIT(BHU)-IIITH system for CoNLL–SIGMORPHON 2018 Shared Task on Universal Morphological Reinflection. The System Description Paper "IIT(BHU)–IIITH at CoNLL–SIGMORPHON 2018 Shared Task on
Universal Morphological Reinflection" can be accessed [here](http://aclweb.org/anthology/K18-3013).

## Abstract of System Description Paper
This   paper   describes   the   systems   submitted  by  IIT  (BHU),  Varanasi/IIIT  Hyderabad (IITBHU–IIITH)   for   Task   1   of   CoNLL–SIGMORPHON  2018  Shared  Task  on  Universal  Morphological  Reinflection  (Cotterell et al., 2018).   The task is to generate the inflected form given a lemma and set of morphological features. The systems are evaluated on over 100 distinct languages and three differentresource settings (low, medium and high). We formulate the task as a sequence to sequence learning problem.   As most of the characters in inflected form are copied from the lemma, we use Pointer-Generator Network (See et al., 2017)  which  makes  it  easier  for  the  system to copy characters from the lemma.  Pointer-Generator Network also helps in dealing with out-of-vocabulary characters during inference. Our best performing system stood 4th among 28  systems,  3rd  among  23  systems  and  4th among  23  systems  for  the  low,  medium  and high resource setting respectively.

## Python packages required
* python-Levenshtein 0.12.0
* PyTorch 0.4

## Instructions
After cloning this repository follow instructions in `data` and `lib` folder to get the required data and libraries.

## Files
* `train.py` produces output for one particular language dataset pair.
* `generate_output.py` generates a complete submission.
