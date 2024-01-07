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

## Training, Evaluation and Prediction
If using GPU: [-gpu] = 0 and [-cuda] = "True", otherwise: [-gpu] = -1

-training
1. training without external knowledge or training the baseline
```bash
python main_run_4jee.py -do_train True -add_candi False -gpu 0 -cuda True
```
2. training with external knowledge 
```bash
python main_run_4jee.py -do_train True -add_candi True -gpu 0 -cuda True -use_SOTA_model True
```


-evaluation
```bash
python main_run_4jee.py -do_eval True -add_candi False -gpu 0 -cuda True -use_SOTA_model True
```

-prediction
1. predict a single sentence or text file
```bash
python main_run_4jee.py -do_test_sihgle True -use_SOTA_model True
```
or 
```bash
python main_run_4jee.py -single_test_file -test_input_file [file name] -use_SOTA_model True
```

2. predict event for entity linking task
```bash
python main_run_4jee.py -do_test_ELdata True -use_SOTA_model True
```



