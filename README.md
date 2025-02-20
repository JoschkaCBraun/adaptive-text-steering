# Controlling Topical Summarization with Steering Vectors

Author: Joschka Braun  
Description: This repository contains the code of my research project, which evaluates the
effectiveness of steering vectors for controlling topical summarization.

## Project Structure

The repository is organized as follows:

```
.
├── config/             # Configuration files for experiments
├── data/               # Results from the experiments
├── datasets/           # Datasets used in the experiments
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

---

## Installation

This project uses **Python 3.12** and **Poetry** for dependency management. Follow these steps to set up the environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/JoschkaCBraun/research-project.git
   cd research-project
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

---

Dev dependencies include `pytest`, `mypy`, `black`, `ruff`, and more, as specified in `pyproject.toml`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.