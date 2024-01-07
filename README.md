# Biomedical Event Extraction (Joint4E-EE) Model For [Joint event extraction and entity linking model](https://arxiv.org/abs/2305.14645)
- A deep leanring framework with BERT and multi-perceptrons to predict named entities, triggers, and events from biomedical texts. The EE model results are reported in our [paper](https://arxiv.org/abs/2305.14645)

## Model Structures
- Based on [Pretrained BERT](https://github.com/allenai/scibert) as encoder.
- Using multi-classification layers to predict entities, triggers and event.
- Leveraging encoded knowledge bases from entity linking model [(Joint4E-EL)](https://github.com/lxc-dolphin/BioJEL).
- Integrating external knowledge information by applying element-wise addition to the representation of each entity.
  
<p align="center">
    <br>
    <img src="https://github.com/lxc-dolphin/BioJEE/blob/main/sup/fig_git_EE.png" width="900"/>
    <br>
<p>

## Tasks
- Joint4E-EE model has been trained and evaluated on the following tasks.
1. GENIA 2011 [(ge11)](http://2011.bionlp-st.org/home/genia-event-extraction-genia)
2. PHAEDRA [(pharmacovigilance)](https://www.nactem.ac.uk/PHAEDRA/)

## Requirements
- Python 3.8
- Pytorch (torch==1.1.0 torchvision==0.3.0, cuda92)
- Install packages

```bash
sh install_EE.sh
```


