# Third-party licenses

This repository contains only its own code. It downloads or imports the following at run time; each
keeps its own license, and using them means following those licenses. Checked 10 Nov 2026.

## Models and data

| What | Used for | License | Source |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | The model every benchmark run trains | Apache-2.0 | https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct |
| Qwen2.5-1.5B-Instruct | The 1.5B check | Apache-2.0 | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct |
| GSM8K (`openai/gsm8k`, `main`) | Training, validation and test problems | MIT | https://huggingface.co/datasets/openai/gsm8k |
| all-MiniLM-L6-v2 | Question embeddings for the BV-Blend clusters (`scripts/make_clusters.py`) | Apache-2.0 | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |

Model weights and the dataset are never committed to this repository. Files derived from them
(the clusters file, completions saved in `results/`) carry their licenses too: GSM8K problems are MIT,
model outputs follow the model's Apache-2.0 terms.

## Libraries

| Library | Version | License |
|---|---|---|
| PyTorch | 2.14.1 | BSD-3-Clause, with parts under other permissive licenses (see its LICENSE) |
| Transformers | 5.19.0 | Apache-2.0 |
| TRL | 1.15.0 | Apache-2.0 |
| Datasets | 5.1 or later | Apache-2.0 |
| PEFT | any | Apache-2.0 |
| huggingface_hub | any | Apache-2.0 |
| NumPy | any | BSD-3-Clause |
| PyYAML | any | MIT |
| scikit-learn | any | BSD-3-Clause |
| SciPy | any | BSD-3-Clause |
| Matplotlib | any | Matplotlib License (PSF-based, permissive) |
| pytest | any | MIT |
| Gymnasium | any | MIT |

## Method papers

The 16 advantage estimators in `src/advantages/` were written from the equations in their papers
(cited in each file and in `docs/method-specs.md`), not copied from the authors' code. Where a
constant or default was taken from an author's released code, the file says so.
