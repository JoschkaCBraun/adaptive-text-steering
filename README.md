# Beyond Multiple Choice: Evaluating Steering Vectors for Adaptive Free-Form Summarization

This repository contains the implementation and experimental results for the paper 
"Beyond Multiple Choice: Evaluating Steering Vectors for Adaptive Free-Form Summarization".

## Paper

This work has been published at the **ICML 2025 Workshop on Reliable and Responsible Foundation Models**.

- **ArXiv**: [https://arxiv.org/abs/2505.24859](https://arxiv.org/abs/2505.24859)
- **OpenReview**: [https://openreview.net/forum?id=sbm53EmmGp](https://openreview.net/forum?id=sbm53EmmGp&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICML.cc%2F2025%2FWorkshop%2FR2-FM%2FAuthors%23your-submissions))
- **PDF**: [icml_2025_r2fm_workshop.pdf](icml_2025_r2fm_workshop.pdf) (included in this repository)

## Project Structure

The repository is organized as follows:

```
.
├── config/             # Configuration files for experiments
├── data/               # Results from the experiments
├── datasets/           # Datasets used in the experiments
│   ├── sentiment/      # Sentiment vectors synthetic training data
│   ├── readability/    # Readability vectors synthetic training data
│   ├── toxicity/       # Toxicity vectors synthetic training data
│   └── topic/          # Topic vectors training representations
├── notebooks/          # Jupyter notebooks for exploratory analysis and visualization
├── scripts/            # Supporting scripts for data preparation and preprocessing
├── src/                # Core source code
│   ├── __init__.py
│   ├── experiments/    # Main steering experiments
│   ├── utils/
│   ├── plot_results.py # plot results for token reweighting
│   ├── run_experiments.py # run token reweighting experiments
│   ├── score_results.py   # score token reweighting experiments
├── tests/              # Tests for some of the steering functionalities
├── pyproject.toml      # Poetry configuration and dependencies
├── poetry.lock         # Locked dependencies
├── LICENSE             # License for the repository
├── README.md           # Project documentation (this file)
```

## Installation

This project uses **Python 3.12** and **Poetry** for dependency management. Follow these steps to set up the environment:

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd [repository-name]
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

## Development

The project includes development dependencies for testing (`pytest`), type checking (`mypy`), 
code formatting (`black`), and linting (`ruff`), as specified in `pyproject.toml`.

## Datasets

The synthetic datasets used for training the steering vectors can be found in the following locations:
- Sentiment data: `datasets/sentiment/`
- Readability data: `datasets/readability/`
- Toxicity data: `datasets/toxicity/`
- Topic representations: `datasets/topic/`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.