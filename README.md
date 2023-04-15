# UniLC - Unified Language Checking

This repo contains the code and access to data used in the paper [Interpretable Unified Language Checking](https://arxiv.org/pdf/2304.03728.pdf).

## Code and data download

The code and data (UniLC benchmark) can be downloaded by
```
git clone https://https://github.com/luohongyin/UniLC.git
cd UniLC/
bash download.sh
```

The evaluation corpora will be saved at `UniLC/ulsc_data/`

## OpenAI API key
Paste your OpenAI API key in the `openai-key.txt` file or replace the corresponding code in `general_check.py` (line 20).

## Reproducing the experiments

The experiments can be reproduced on four tasks:

- Fact checking
    - [Climate](https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html): `climate`
    - [PubHealth](https://www.aclweb.org/anthology/2020.emnlp-main.623): `health`
- Hate speech
    - [Hate speech detection](https://www.aclweb.org/anthology/W18-5102): `hsd`
- Stereotypes
    - [Social bias frame](https://maartensap.com/social-bias-frames/): `sbic`

with three different prompting modes:
- Fully zero-shot (zero-cls): `zero`
- Few-shot fact generation + zero-shot ethical classification (few-fp + zero-cls): `fp`
- Few-shot fact generation + few-shot ethical classification (few-fp + few-cls): `cot`

An experiment can be ran with
```
usage: python general_check.py [-h] [-t TASK] [-m MODE] [-s START_IDX] [-n EXP_NAME] [-v]

Unified language safety checking with LLMs.

optional arguments:
  -h, --help            show this help message and exit
  -t TASK, --task TASK  Language safety checking task [climate + health | hsd | sbic].
  -m MODE, --mode MODE  Prompting mode [zero | fp | cot].
  -s START_IDX, --start-idx START_IDX
                        Index of the first sample to process.
  -n EXP_NAME, --exp-name EXP_NAME
                        Name tag for the experiment log file.
  -v, --verbose
```

If a full evaluation is supposed to be conducted, set `args.start_idx = 0, args.verbose = False`. To look into the model behavior on the `i`-th test sample, set `args.start_idx = i, args.verbose = True`

## Citation

Please cite our paper if our code and data are helpful!
```
@article{zhang2023interpretable,
  title={Interpretable Unified Language Safety Checking},
  author={Zhang, Tianhua and Luo, Hongyin and Chuang, Yung-Sung and Fang, Wei and Gaitskell, Luc and Hartvigsen, Thomas and Wu, Xixin and Fox, Danny and Meng, Helen and Glass, James},
  journal={arXiv preprint arXiv:2304.03728},
  year={2023}
}
```

## Support and Contact

If there is any question, please submit an issue or contact:
- Hongyin Luo: hyluo AT mit DOT edu
- Tianhua Zhang: thzhang AT link DOT cuhk DOT edu DOT hk
