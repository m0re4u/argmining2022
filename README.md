# CLTL Team @ ArgMining 2022 Shared Task

See description of the task [here](https://phhei.github.io/ArgsValidNovel/).

We are a team of awesome [CLTL](http://www.cltl.nl/) PhDs:
- [Michiel van der Meer](https://liacs.leidenuniv.nl/~meermtvander/)
- [Myrthe Reuver](https://myrthereuver.github.io/)
- [Urja Khurana](https://urjakh.github.io/)
- [Lea Krause](https://lkra.github.io/)
- [Selene Báez Santamaría](https://selbaez.github.io/)


## Installation
Install packages with `pip install -r requirements.txt`. This file may or may not be complete/up-to-date..

## Wandb integration
See [here](https://docs.wandb.ai/guides/integrations/huggingface) on how to set it up. For convenience, I'm using python-dotenv for creating the environment variable for the wandb project name
1. Create a `.env` file in project root.
2. Create following variable e.g. `WANDB_PROJECT=argmining2022-mtl`


## Usage

### GPT3 models

First, please create an account with [OpenAI](https://auth0.openai.com/u/signup). Get your API key and set it as an environmental variable. For running the *best* approach using GPT3 (for example on the test set), run:

```bash
python prompting.py --n_shot 4 --prompt_style 5
```

To evaluate a file with predictions, you can do:

```python
from baseline import evaluate_from_file

evaluate_from_file("predictions/dump_prompt_5_results.json")
```


### MTL training
See the arguments in `train_mtl.py`.
