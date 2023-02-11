import os
import random
from collections import deque
from pathlib import Path

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf

import kitchen_env
import wandb


if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)


@hydra.main(config_path=".", config_name="train", version_base="1.2")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)
    train_data, test_data = hydra.utils.instantiate(cfg.data)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.batch_size, shuffle=True, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=cfg.batch_size, shuffle=False, pin_memory=True
    )
    cbet_model = hydra.utils.instantiate(cfg.model).to(cfg.device)
    if cfg.load_path:
        cbet_model.load_model(Path(cfg.load_path))
    optimizer = cbet_model.configure_optimizers(
        weight_decay=cfg.optim.weight_decay,
        learning_rate=cfg.optim.lr,
        betas=cfg.optim.betas,
    )
    goal_fn = hydra.utils.instantiate(cfg.goal_fn)
    env = hydra.utils.instantiate(cfg.env.gym)
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    run_name = run.name or "Offline"
    save_path = Path(cfg.save_path) / run_name
    save_path.mkdir(parents=True, exist_ok=False)

    @torch.no_grad()
    def eval_on_env(cfg, num_evals=cfg.num_env_evals, num_eval_per_goal=1):
        avg_reward = 0
        for goal_idx in range(num_evals):
            for _ in range(num_eval_per_goal):
                obs_stack = deque(maxlen=cfg.eval_window_size)
                obs_stack.append(env.reset())
                done, step, total_reward = False, 0, 0
                goal, _ = goal_fn(env, obs_stack[-1], goal_idx, step)
                while not done:
                    obs = torch.from_numpy(np.stack(obs_stack)).float().to(cfg.device)
                    goal = torch.as_tensor(goal, dtype=torch.float32, device=cfg.device)
                    action, _, _ = cbet_model(obs.unsqueeze(0), goal.unsqueeze(0), None)
                    obs, reward, done, info = env.step(action[0, -1, :].cpu().numpy())
                    step += 1
                    total_reward += reward
                    obs_stack.append(obs)
                    goal, _ = goal_fn(env, obs_stack[-1], goal_idx, step)
                avg_reward += total_reward
        return avg_reward / (num_evals * num_eval_per_goal)

    for epoch in tqdm.trange(cfg.epochs):
        cbet_model.eval()
        if epoch % cfg.eval_on_env_freq == 0:
            avg_reward = eval_on_env(cfg)
            wandb.log({"eval_on_env": avg_reward})

        if epoch % cfg.eval_freq == 0:
            total_loss = 0
            with torch.no_grad():
                for data in test_loader:
                    obs, act, goal = (x.to(cfg.device) for x in data)
                    _, loss, loss_dict = cbet_model(obs, goal, act)
                    total_loss += loss.item()
                    wandb.log({"eval/{}".format(x): y for (x, y) in loss_dict.items()})
            print(f"Test loss: {total_loss / len(test_loader)}")

        for data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            obs, act, goal = (x.to(cfg.device) for x in data)
            _, loss, loss_dict = cbet_model(obs, goal, act)
            wandb.log({"train/{}".format(x): y for (x, y) in loss_dict.items()})
            loss.backward()
            optimizer.step()

        if epoch % cfg.save_every == 0:
            cbet_model.save_model(save_path)

    avg_reward = eval_on_env(
        cfg,
        num_evals=cfg.num_final_evals,
        num_eval_per_goal=cfg.num_final_eval_per_goal,
    )
    wandb.log({"final_eval_on_env": avg_reward})
    return avg_reward


if __name__ == "__main__":
    main()
