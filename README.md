# Time-Transformer

<p align="center">
<img src=imgs/timetransformer.png />
</p>

This is a Python implementation of the paper "[**Time-Transformer: Integrating Local and Global Features for Better Time Series Generation**](https://arxiv.org/abs/2312.11714)" (SDM24).

The original source code can be found at https://github.com/Lysarthas/Time-Transformer.git
This is forked version of it and following changes have been made:
1) Jupyter Notebook "**StartHere**" is added which provides the sequence of code in order to run it in local host.
2) Versions of certain imports have been updated
3) Comments are added in networks.py,aae.py etc. to provide useful explanation to the code

Jupyter Notebook "**tutorial**" provide a tutorial for training and evaluating with different metrics (using "**sine_cpx**" dataset). FID score are calculated with "**fid_score**" in `ts2vec`, directly using model "[**TS2Vec**](https://github.com/yuezhihan/ts2vec)".

If you find this model useful and put it in your publication, we encourage you to add the following references:
```bibtex
@misc{liu2023timetransformer,
      title={Time-Transformer: Integrating Local and Global Features for Better Time Series Generation}, 
      author={Yuansan Liu and Sudanthi Wijewickrema and Ang Li and Christofer Bester and Stephen O'Leary and James Bailey},
      year={2023},
      eprint={2312.11714},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
