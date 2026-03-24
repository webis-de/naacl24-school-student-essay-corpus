# School Student Essay Corpus

This repository contains the corpus and code for the paper "A School Student Essay Corpus for Analyzing Interactions of Argumentative Structure and Quality", as accepted to the NAACL 2024 conference.

[https://aclanthology.org/2024.naacl-long.145/](https://aclanthology.org/2024.naacl-long.145/)

## Version 2
The paper used the first version of our annotations (arg-school-corpus-annotations.json). However, we later found some contradictory annotations, that are now removed in version 2 of our corpus (arg-school-corpus-annotations-v2.json) 

#### How to build corpus-v2

- download `arg-school-corpus-annotations-v2.json` and `corpusbuilder_v2.py`
- register at [https://fd-lex.uni-koeln.de/](https://fd-lex.uni-koeln.de/)
- download "Scriptoria" corpus transcripts and datatable
- save transcripts (8 pdf files) and datatable (one xlsx file) in ```.\transcripts``` directory (if needed paths can be adjusted at the top of the `corpusbuilder_v2.py` file)
- install dependencies ```pip install pypdfium2 pandas```
- execute ```python corpusbuilder_v2.py```
- combined corpus is saved as `arg-school-corpus-created.json`

## Reference
If you use our code or data, please cite the work as follows: 

Maja Stahl, Nadine Michel, Sebastian Kilsbach, Julian Schmidtke, Sara Rezat, and Henning Wachsmuth. 2024. A School Student Essay Corpus for Analyzing Interactions of Argumentative Structure and Quality. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 2661–2674, Mexico City, Mexico. Association for Computational Linguistics.

## Bibtex

```
@inproceedings{stahl-etal-2024-school,
    title = "A School Student Essay Corpus for Analyzing Interactions of Argumentative Structure and Quality",
    author = "Stahl, Maja  and
      Michel, Nadine  and
      Kilsbach, Sebastian  and
      Schmidtke, Julian  and
      Rezat, Sara  and
      Wachsmuth, Henning",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.145/",
    doi = "10.18653/v1/2024.naacl-long.145",
    pages = "2661--2674"
}
```
