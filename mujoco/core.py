import math

import utils
import torch
import numpy as np
import gymnasium as gym
from dotmap import DotMap
from omegaconf import OmegaConf
from model import DecisionTransformer
from hydra.utils import instantiate
from buffer import SequenceBuffer
import torch.nn.functional as F
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics
from drop_fn import DropWrapper, PerFeatureDropWrapper
import csv
from datetime import datetime
import pandas as pd
from itertools import combinations
from peft import LoraConfig, get_peft_model, LoraModel
import os
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_perf_drop_curve(env: gym.vector.Env, model, rtg_target, drop_ps:list, seed):
    return_means = []
    use_per_feat = getattr(model, 'use_per_feat_dropping', False)
    for drop_p in drop_ps:
        if use_per_feat:
            drop_env = PerFeatureDropWrapper(env, drop_p, model.drop_features, seed)
        else:
            drop_env = DropWrapper(env, drop_p, seed)
        mean, _, _, _, _ = eval(drop_env, model, rtg_target)
        return_means.append(mean)
    return return_means

def get_perf_drop_csv(env: gym.vector.Env, model, rtg_target, drop_ps:list, seed, modelname):
    droplist = []
    use_per_feat = getattr(model, 'use_per_feat_dropping', False)
    for drop_p in drop_ps:
        if use_per_feat:
            drop_env = PerFeatureDropWrapper(env, drop_p, model.drop_features, seed)
        else:
            drop_env = DropWrapper(env, drop_p, seed)
        mean, std, q75, q25, returns = eval(drop_env, model, rtg_target)
        droplist.append(returns)
    with open(modelname+'seed'+str(seed)+'.csv','w') as f:
        writer = csv.writer(f)
        for row in droplist:
            writer.writerow(row)

def apply_feature_lora(model, per_feat_dropstep: np.ndarray) -> None:
    """現在 drop されている特徴量に対応するアダプタをマージしてモデルに適用する。

    Args:
        model: PEFT でラップされた DecisionTransformer
        per_feat_dropstep: shape (n_feats,)。各特徴量の dropstep 距離
    """
    # 現在 drop されている特徴量の state インデックス集合
    active_set = frozenset(
        model.drop_features[i]
        for i, ds in enumerate(per_feat_dropstep)
        if ds > 0
    )

    # 前回呼び出しと同じ組み合わせなら何もしない
    if active_set == model._feature_lora_active_set:
        return
    model._feature_lora_active_set = active_set

    if not active_set:
        # どの特徴量も drop されていない → LoRA を無効化してベースモデルで推論
        model.disable_adapter_layers()
        return

    # LoRA 層が無効化されていた場合は再有効化
    model.enable_adapter_layers()

    cache_key = tuple(sorted(active_set))
    adapter_name = "merged_" + "_".join(map(str, cache_key))

    if adapter_name not in model._feature_lora_cache:
        source_adapters = [f"feat_{idx}" for idx in cache_key]
        n = len(source_adapters)
        w = model._feature_lora_merge_weight
        model.add_weighted_adapter(
            adapters=source_adapters,
            weights=[w] * n,
            adapter_name=adapter_name,
            combination_type="linear",
        )
        model._feature_lora_cache.add(adapter_name)

    model.set_adapter(adapter_name)


def save_lora_model(model, task_name, seed, save_path, drop_p=None, drop_features=None):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    feats_str = "feats" + "-".join(map(str, drop_features)) if drop_features else "featsnone"
    path = f"{save_path}/{task_name}_{drop_p}_{feats_str}_seed_{seed}_{timestamp}"
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    print(f"LoRA adapter saved to {path}")
@torch.no_grad()
def eval(env: gym.vector.Env, model: DecisionTransformer, rtg_target):
    # parallel evaluation with vectorized environment
    model.eval()
    episodes = env.num_envs
    reward, returns = np.zeros(episodes), np.zeros(episodes)
    done_flags = np.zeros(episodes, dtype=np.bool8)

    state_dim = utils.get_space_shape(env.observation_space, is_vector_env=True)
    act_dim = utils.get_space_shape(env.action_space, is_vector_env=True)
    max_timestep = model.max_timestep
    context_len = model.context_len
    timesteps = torch.arange(max_timestep, device=device)

    use_per_feat = getattr(model, 'use_per_feat_dropping', False)
    if use_per_feat:
        n_feats = len(model.drop_features)
        dropsteps = torch.zeros(max_timestep, n_feats, device=device, dtype=torch.long)
        per_feat_dropstep = np.zeros(n_feats, dtype=np.int32)
    else:
        dropsteps = torch.zeros(max_timestep, device=device, dtype=torch.long)
        dropstep = 0

    state, _ = env.reset(seed=[np.random.randint(0, 10000) for _ in range(episodes)])

    states = torch.zeros((episodes, max_timestep, state_dim), dtype=torch.float32, device=device)
    actions = torch.zeros((episodes, max_timestep, act_dim), dtype=torch.float32, device=device)
    rewards_to_go = torch.zeros((episodes, max_timestep, 1), dtype=torch.float32, device=device)

    reward_to_go, timestep = rtg_target, 0
    while not done_flags.all():
        states[:, timestep] = torch.from_numpy(state).to(device)
        rewards_to_go[:, timestep] = reward_to_go - torch.from_numpy(returns).to(device).unsqueeze(-1)

        if use_per_feat:
            dropsteps[timestep] = torch.from_numpy(per_feat_dropstep).to(device)
        else:
            dropsteps[timestep] = dropstep

        obs_index = torch.arange(max(0, timestep-context_len+1), timestep+1)

        # 動的にLoRAを変化させるためにドロップ率を計算
        if model.dynamic_lora:
            drop_rate = np.count_nonzero(dropsteps.cpu().numpy()) / dropsteps.numel()
            if drop_rate<0.1:
                model.set_adapter("0_0")
            elif drop_rate<0.2:
                model.set_adapter("0_1")
            elif drop_rate<0.3:
                model.set_adapter("0_2")
            elif drop_rate<0.4:
                model.set_adapter("0_3")
            elif drop_rate<0.5:
                model.set_adapter("0_4")
            elif drop_rate<0.6:
                model.set_adapter("0_5")
            elif drop_rate<0.7:
                model.set_adapter("0_6")
            elif drop_rate<0.8:
                model.set_adapter("0_7")
            elif drop_rate<0.9:
                model.set_adapter("0_8")
            else:
                model.set_adapter("0_9")
            print(f"Drop rate: {drop_rate:.2f}, set LoRA adapter to {model.active_adapter}")

        if use_per_feat and getattr(model, 'use_feature_lora', False):
            apply_feature_lora(model, per_feat_dropstep)

        if use_per_feat:
            ds = dropsteps[None, obs_index]  # (1, T, n_feats)
            _, action_preds, _ = model.forward(
                states[:, obs_index],
                actions[:, obs_index],
                rewards_to_go[:, obs_index],          # no rtg shift for per-feature eval
                timesteps[None, obs_index],
                ds)
        else:
            _, action_preds, _ = model.forward(
                states[:, obs_index],
                actions[:, obs_index],
                rewards_to_go[:, obs_index - dropsteps[obs_index].cpu()],  # drop rewards
                timesteps[None, obs_index],
                dropsteps[None, obs_index])

        action = action_preds[:, -1].detach()
        actions[:, timestep] = action

        state, reward, dones, truncs, info = env.step(action.cpu().numpy())

        if use_per_feat:
            feat_ds = info.get('feat_dropsteps', None)
            per_feat_dropstep = feat_ds if feat_ds is not None else np.zeros(n_feats, dtype=np.int32)
        else:
            dropstep = dropsteps[timestep].item() + 1 if info.get('dropped', False) else 0

        returns += reward * ~done_flags
        done_flags = np.bitwise_or(np.bitwise_or(done_flags, dones), truncs)
        timestep += 1
        q75, q25 = np.percentile(returns, [75, 25])

    return np.mean(returns), np.std(returns), q75, q25, returns


def train(cfg, seed, log_dict, idx, logger, barrier, buffer_dir):
    using_mp = barrier is not None
    utils.config_logging("main_mp.log" if using_mp else "main.log")
    env_name = cfg.env.env_name
    eval_env = gym.vector.make(env_name + '-v4', render_mode="rgb_array", num_envs=cfg.train.eval_episodes, asynchronous=False, wrappers=RecordEpisodeStatistics)
    utils.set_seed_everywhere(eval_env, seed)

    state_dim = utils.get_space_shape(eval_env.observation_space, is_vector_env=True)
    action_dim = utils.get_space_shape(eval_env.action_space, is_vector_env=True)
    drop_cfg = DotMap(OmegaConf.to_container(cfg.buffer.drop_cfg, resolve=True))
    if cfg.part_dropping.use_part_dropping:
        drop_cfg.drop_fn = 'per_feat_const'
        drop_cfg.drop_features = list(cfg.part_dropping.drop_features)
        drop_cfg.drop_p = cfg.part_dropping.drop_p
    buffer = instantiate(cfg.buffer, root_dir=buffer_dir, drop_cfg=drop_cfg, seed=seed)
    model = instantiate(cfg.model, state_dim=state_dim, action_dim=action_dim, action_space=eval_env.envs[0].action_space, state_mean=buffer.state_mean, state_std=buffer.state_std, device=device)
    print(model)
    if cfg.load_model.use_load_model:
        model.load(cfg.load_model.load_model_path)
        logger.info(f"Loaded model from {cfg.load_model.load_model_path}")
    if cfg.lora.use_lora:
        peft_config = LoraConfig(
            task_type=None,
            inference_mode=False,
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            target_modules=["q_net", "v_net"],
        )
        model = get_peft_model(model, peft_config)
        model.r=cfg.lora.r
        model.lora_alpha=cfg.lora.lora_alpha
        model.lora_dropout=cfg.lora.lora_dropout
        model.print_trainable_parameters()
        using_lora = True
        lora_save_path = cfg.lora.lora_save_path
    else:
        using_lora = False
        lora_save_path = None
    if cfg.dynamic_lora.use_dynamic_lora:
        logger.info("Using dynamic LoRA with drop-aware training")
        model.load_adapter(cfg.dynamic_lora.lora_load_path.drop0_0, adapter_name="0_0",is_trainable=True)
        model.load_adapter(cfg.dynamic_lora.lora_load_path.drop0_1, adapter_name="0_1",is_trainable=True)
        model.load_adapter(cfg.dynamic_lora.lora_load_path.drop0_2, adapter_name="0_2",is_trainable=True)
        model.load_adapter(cfg.dynamic_lora.lora_load_path.drop0_3, adapter_name="0_3",is_trainable=True)
        model.load_adapter(cfg.dynamic_lora.lora_load_path.drop0_4, adapter_name="0_4",is_trainable=True)
        model.load_adapter(cfg.dynamic_lora.lora_load_path.drop0_5, adapter_name="0_5",is_trainable=True)
        model.load_adapter(cfg.dynamic_lora.lora_load_path.drop0_6, adapter_name="0_6",is_trainable=True)
        model.load_adapter(cfg.dynamic_lora.lora_load_path.drop0_7, adapter_name="0_7",is_trainable=True)
        model.load_adapter(cfg.dynamic_lora.lora_load_path.drop0_8, adapter_name="0_8",is_trainable=True)
        model.load_adapter(cfg.dynamic_lora.lora_load_path.drop0_9, adapter_name="0_9",is_trainable=True)
        model.train()
        model.dynamic_lora = True
        model.load_path = cfg.load_model.load_model_path
    else:
        model.dynamic_lora = False
    print("Model architecture:")
    print(model)
    if cfg.part_dropping.use_part_dropping:
        logger.info(f"Using per-feature dropping: features={list(cfg.part_dropping.drop_features)}, drop_p={cfg.part_dropping.drop_p}")
        model.use_per_feat_dropping = True
        model.drop_features = list(cfg.part_dropping.drop_features)
    else:
        model.use_per_feat_dropping = False
    if cfg.feature_lora.use_feature_lora:
        logger.info("Loading feature-specific LoRA adapters")
        adapter_paths = OmegaConf.to_container(cfg.feature_lora.adapter_paths, resolve=True)
        for adapter_name, path in adapter_paths.items():
            model.load_adapter(path, adapter_name=adapter_name, is_trainable=False)
            logger.info(f"Loaded adapter '{adapter_name}' from {path}")
        model.use_feature_lora = True
        model._feature_lora_cache = set()
        model._feature_lora_active_set = None
        model._feature_lora_merge_weight = cfg.feature_lora.merge_weight
        if cfg.feature_lora.precompute_all:
            drop_features = list(cfg.part_dropping.drop_features)
            logger.info(f"Pre-computing all {2**len(drop_features) - 1} feature LoRA combinations")
            for r in range(1, len(drop_features) + 1):
                for combo in combinations(drop_features, r):
                    cache_key = tuple(sorted(combo))
                    merged_name = "merged_" + "_".join(map(str, cache_key))
                    source_adapters = [f"feat_{idx}" for idx in cache_key]
                    n = len(source_adapters)
                    w = cfg.feature_lora.merge_weight
                    model.add_weighted_adapter(
                        adapters=source_adapters,
                        weights=[w] * n,
                        adapter_name=merged_name,
                        combination_type="linear",
                    )
                    model._feature_lora_cache.add(merged_name)
                    logger.info(f"Pre-computed '{merged_name}'")
    else:
        model.use_feature_lora = False
    cfg = DotMap(OmegaConf.to_container(cfg.train, resolve=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step+1)/cfg.warmup_steps, 1))

    logger.info(f"Training seed {seed} for {cfg.train_steps} timesteps with {env_name} {buffer.dataset.title()} dataset")

    
    if using_mp:
        local_log_dict = {key: [] for key in log_dict.keys()}
    else:
        local_log_dict = log_dict
        for key in local_log_dict.keys():
            local_log_dict[key].append([])

    best_reward = -np.inf
    utils.write_to_dict(local_log_dict, 'rtg_target', cfg.rtg_target, using_mp)
    for timestep in range(1, cfg.train_steps + cfg.finetune_steps + 1):
        states, actions, rewards_to_go, timesteps, dropsteps, mask = buffer.sample(cfg.batch_size)
        # no need for attention mask for the model as we always pad on the right side, whose attention is ignored by the casual mask anyway
        state_preds, action_preds, return_preds = model.forward(states, actions, rewards_to_go, timesteps, dropsteps)
        action_preds = action_preds[mask]
        action_loss = F.mse_loss(action_preds, actions[mask].detach(), reduction='mean')
        utils.write_to_dict(local_log_dict, 'action_loss', action_loss.item(), using_mp)

        optimizer.zero_grad()
        action_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        scheduler.step()

        if timestep % cfg.eval_interval == 0:
            eval_mean, eval_std,_,_,_ = eval(eval_env, model, cfg.rtg_target)
            utils.write_to_dict(local_log_dict, 'eval_steps', timestep - 1, using_mp)
            utils.write_to_dict(local_log_dict, 'eval_returns', eval_mean, using_mp)
            d4rl_score = utils.get_d4rl_normalized_score(env_name, eval_mean)
            utils.write_to_dict(local_log_dict, 'd4rl_score', d4rl_score, using_mp)
            logger.info(f"Seed: {seed}, Step: {timestep}, Eval mean: {eval_mean:.2f}, Eval std: {eval_std:.2f}")

            if eval_mean > best_reward:
                best_reward = eval_mean
                model.save(f'best_train_seed_{seed}' if timestep <= cfg.train_steps else f'best_finetune_seed_{seed}')
                logger.info(f'Seed: {seed}, Save best model at eval mean {best_reward:.4f} and step {timestep}')

        if timestep % cfg.plot_interval == 0:
            utils.sync_and_visualize(log_dict, local_log_dict, barrier, idx, timestep, f'{env_name} {buffer.dataset.title()}', using_mp)

        if timestep == cfg.train_steps:
            model.save(f'final_train_seed_{seed}')
            model.load(f'best_train_seed_{seed}')

            perf_drop_curve = get_perf_drop_curve(eval_env, model, cfg.rtg_target, cfg.eval_drop_ps, seed)
            get_perf_drop_csv(eval_env, model, cfg.rtg_target, cfg.eval_drop_ps, seed,'train')
            for drop_perf in perf_drop_curve:
                utils.write_to_dict(local_log_dict, 'perf_drop_train', drop_perf, using_mp)
            if cfg.finetune_steps > 0 and model.drop_aware:
                logger.info(f"Finetuning seed {seed} for {cfg.finetune_steps} timesteps with {env_name} {buffer.dataset.title()} dataset")
                model.freeze_trunk()
                buffer.drop_fn.drop_p = drop_cfg.finetune_drop_p
                buffer.drop_fn.update_dropmask()
                best_reward = -np.inf # ensure we will save best finetune model at least once
    
    if cfg.finetune_steps > 0 and model.drop_aware:
        model.save(f'final_finetune_seed_{seed}')
        model.load(f'best_finetune_seed_{seed}')

        perf_drop_curve = get_perf_drop_curve(eval_env, model, cfg.rtg_target, cfg.eval_drop_ps, seed)
        get_perf_drop_csv(eval_env, model, cfg.rtg_target, cfg.eval_drop_ps, seed,'finetuned')
        for drop_perf in perf_drop_curve:
            utils.write_to_dict(local_log_dict, 'perf_drop_finetune', drop_perf, using_mp)

    utils.sync_and_visualize(log_dict, local_log_dict, barrier, idx, timestep, f'{env_name} {buffer.dataset.title()}', using_mp)
    logger.info(f"Finish training seed {seed} with everage eval mean: {eval_mean}")
    if using_lora and lora_save_path is not None:
        drop_features = drop_cfg.drop_features if drop_cfg.drop_fn == 'per_feat_const' else None
        save_lora_model(model, env_name, seed, lora_save_path, drop_p=drop_cfg.drop_p, drop_features=drop_features)
    return eval_mean