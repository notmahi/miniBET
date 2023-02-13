# miniBET
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch version test](https://github.com/notmahi/miniBET/workflows/PyTorch%20version%20tests/badge.svg)](https://github.com/pytorch/ignite/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Clean implementation of [conditional](https://play-to-policy.github.io) and [unconditional behavior transformer](https://mahis.life/bet). The API is heavily inspired by [Lucidrains' implementations](https://github.com/lucidrains/), and the implementation is heavily indebted to Andrej Karpathy's implementation of [NanoGPT](https://github.com/karpathy/nanoGPT).


## Installation
```bash
git clone git@github.com:notmahi/miniBET.git
cd miniBET
pip install --upgrade .
```

## Usage
```python
import torch
from behavior_transformer import BehaviorTransformer, GPT, GPTConfig

conditional = True
obs_dim = 50
act_dim = 8
goal_dim = 50 if conditional else 0
K = 32
T = 16
batch_size = 256

cbet = BehaviorTransformer(
    obs_dim=obs_dim,
    act_dim=act_dim,
    goal_dim=goal_dim,
    gpt_model=GPT(
        GPTConfig(
            block_size=144,
            input_dim=obs_dim,
            n_layer=6,
            n_head=8,
            n_embd=256,
        )
    ),  # The sequence model to use.
    n_clusters=K,  # Number of clusters to use for k-means discretization.
    kmeans_fit_steps=5,  # The k-means discretization is done on the actions seen in the first kmeans_fit_steps.
)

optimizer = cbet.configure_optimizers(
    weight_decay=2e-4,
    learning_rate=1e-5,
    betas=[0.9, 0.999],
)

for i in range(10):
    obs_seq = torch.randn(batch_size, T, obs_dim)
    goal_seq = torch.randn(batch_size, T, goal_dim)
    action_seq = torch.randn(batch_size, T, act_dim)
    if i <= 7:
        # Training.
        train_action, train_loss, train_loss_dict = cbet(obs_seq, goal_seq, action_seq)
    else:
        # Action inference
        eval_action, eval_loss, eval_loss_dict = cbet(obs_seq, goal_seq, None)
```

If you want to use your own sequence model, you can pass in that model as the `gpt_model` argument in the `BehaviorTransformer` constructor. The only extra requirement for the sequence model (beyond being a subclass of `nn.Module` having the input and output of the right shape) is to have a `configure_optimizer` method that takes in the `weight_decay`, `learning_rate`, and `betas` arguments and returns a `torch.optim.Optimizer` object.

## Example task
Try out the example task on the [Franka kitchen](https://robotics.farama.org/envs/franka_kitchen/) environment. You will need to install extra requirements that you can find in the `examples/requirements-dev.txt` file.

Fill out the details in `examples/train.yaml` with the paths of your downloaded dataset from [here](https://osf.io/983qz/).

If you have installed all the dependencies and have downloaded the dataset, you can run the example with:
```bash
cd examples
python train.py
```
It should take about 50 minutes to train on a single GPU, and have a final performance of ~3.2 conditioned tasks on average.

## Citation
If you use this code in your research, please cite the following papers whenever appropriate:

```
@inproceedings{
    shafiullah2022behavior,
    title={Behavior Transformers: Cloning $k$ modes with one stone},
    author={Nur Muhammad Mahi Shafiullah and Zichen Jeff Cui and Ariuntuya Altanzaya and Lerrel Pinto},
    booktitle={Advances in Neural Information Processing Systems},
    editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
    year={2022},
    url={https://openreview.net/forum?id=agTr-vRQsa}
}

@article{cui2022play,
    title={From play to policy: Conditional behavior generation from uncurated robot data},
    author={Cui, Zichen Jeff and Wang, Yibin and Shafiullah, Nur Muhammad Mahi and Pinto, Lerrel},
    journal={arXiv preprint arXiv:2210.10047},
    year={2022}
}
```