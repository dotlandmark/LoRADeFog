import numpy as np
import gymnasium as gym

def get_drop_fn(drop_cfg, buffer_size, traj_sp, rng):
    if drop_cfg.drop_fn == 'const':
        return ConstFn(buffer_size, drop_cfg.drop_p, drop_cfg.update_interval, traj_sp, rng)
    elif drop_cfg.drop_fn == 'linear':
        return LinearFn(buffer_size, drop_cfg.start_p, drop_cfg.end_p, drop_cfg.ascend_steps, drop_cfg.update_interval, traj_sp, rng)
    elif drop_cfg.drop_fn == 'per_feat_const':
        return PerFeatureConstFn(buffer_size, list(drop_cfg.drop_features), drop_cfg.drop_p, drop_cfg.update_interval, traj_sp, rng)
    else:
        raise NotImplementedError(f'Unknown drop_fn: {drop_cfg.drop_fn}')


class DropWrapper(gym.Wrapper):
    def __init__(self, env, drop_p, seed) -> None:
        super().__init__(env)
        self.env = env
        self.obs_drop_p = drop_p
        self.last_obs = None
        self.rng = np.random.default_rng(seed)
    
    def step(self, action):
        next_state, reward, done, trunc, info = self.env.step(action)
        if self.rng.random() > self.obs_drop_p: # current observation is not dropped
            self.last_obs = next_state
            info['dropped'] = False
        else:
            info['dropped'] = True
        # the drop of reward is handled in the eval function
        return self.last_obs, reward, done, trunc, info
    
    def reset(self, seed):
        self.last_obs, info = self.env.reset(seed=seed)
        return self.last_obs, info


class DropFn:
    def __init__(self, size, update_interval, traj_sp, rng:np.random.Generator, drop_aware=True) -> None:
        self.size = size
        self.step_count = 0
        self.traj_sp = np.append(traj_sp, size - 1)
        self.dropmask = np.ones((size,), dtype=np.bool8)
        self.dropstep = np.zeros((size,), dtype=np.int32)
        self.update_interval = update_interval
        self.rng = rng
        self.drop_aware = drop_aware
        
    def get_dropsteps(self, selected_index):
        return self.dropstep[selected_index]
    
    def get_dropmasks(self, selected_index):
        return self.dropmask[selected_index]

    def get_traj_sp_ep(self, selected_index):
        sps = max(np.searchsorted(self.traj_sp, selected_index), 1)
        return self.traj_sp[sps - 1], self.traj_sp[sps]

    def step(self):
        if not self.step_count % self.update_interval and self.drop_aware:
            self.update_dropmask()
            self.update_dropstep()
        self.step_count += 1
        
    def update_dropmask(self):
        raise NotImplementedError

    def update_dropstep(self): # get the distance since last valid frame
        # inspired by https://stackoverflow.com/questions/18196811/cumsum-reset-at-nan
        v = np.ones(self.size, dtype=np.int32)
        c = np.cumsum(~self.dropmask)
        d = np.diff(np.concatenate(([0], c[self.dropmask])))
        v[self.dropmask] = -d
        self.dropstep = np.cumsum(v)
        self.dropstep[-1] = 0


class ConstFn(DropFn):
    def __init__(self, size, drop_p, update_interval, traj_sp, rng, drop_aware=True) -> None:
        super().__init__(size, update_interval, traj_sp, rng, drop_aware)
        self.drop_p = drop_p

    def update_dropmask(self):
        self.dropmask = self.rng.random(self.size) > self.drop_p
        self.dropmask[self.traj_sp] = True
        self.dropmask[-1] = False


class LinearFn(DropFn):
    def __init__(self, size, start_p, end_p, ascend_steps, update_interval, traj_sp, rng, drop_aware=True) -> None:
        super().__init__(size, update_interval, traj_sp, rng, drop_aware)
        # assert end_p > start_p, 'drop_p should ascend gradually'
        self.start_p = start_p
        self.end_p = end_p
        self.ascend_steps = ascend_steps

    def update_dropmask(self):
        drop_p = self.end_p * np.min([1, self.step_count / self.ascend_steps]) + \
            self.start_p * max([0, 1 - self.step_count / self.ascend_steps])
        if self.step_count / self.ascend_steps in [0.25, 0.5, 0.75]:
            print('*' * 20 + ' current drop_p is:%g ' % drop_p + '*' * 20)
        self.dropmask = self.rng.random(self.size) > drop_p
        self.dropmask[self.traj_sp] = True
        self.dropmask[-1] = False


class PerFeatureDropFn:
    """Base class for per-feature independent dropping.
    dropmask/dropstep are 2D: (buffer_size, n_feats).
    Each feature in drop_features drops independently.
    """
    def __init__(self, size, drop_features, update_interval, traj_sp, rng, drop_aware=True):
        self.size = size
        self.drop_features = drop_features
        self.n_feats = len(drop_features)
        self.step_count = 0
        self.traj_sp = np.append(traj_sp, size - 1)
        self.dropmask = np.ones((size, self.n_feats), dtype=bool)   # True = valid
        self.dropstep = np.zeros((size, self.n_feats), dtype=np.int32)
        self.update_interval = update_interval
        self.rng = rng
        self.drop_aware = drop_aware

    def get_dropsteps(self, selected_index):
        # selected_index: (B, T) → returns (B, T, n_feats)
        return self.dropstep[selected_index]

    def get_dropmasks(self, selected_index):
        return self.dropmask[selected_index]

    def step(self):
        if not self.step_count % self.update_interval and self.drop_aware:
            self.update_dropmask()
            self.update_dropstep()
        self.step_count += 1

    def update_dropstep(self):
        for f in range(self.n_feats):
            v = np.ones(self.size, dtype=np.int32)
            mask_f = self.dropmask[:, f]
            c = np.cumsum(~mask_f)
            valid_c = c[mask_f]
            if len(valid_c) > 0:
                d = np.diff(np.concatenate(([0], valid_c)))
                v[mask_f] = -d
            self.dropstep[:, f] = np.cumsum(v)
        self.dropstep[-1, :] = 0

    def update_dropmask(self):
        raise NotImplementedError


class PerFeatureConstFn(PerFeatureDropFn):
    """Each feature in drop_features drops independently with constant probability drop_p."""
    def __init__(self, size, drop_features, drop_p, update_interval, traj_sp, rng, drop_aware=True):
        super().__init__(size, drop_features, update_interval, traj_sp, rng, drop_aware)
        self.drop_p = drop_p

    def update_dropmask(self):
        self.dropmask = self.rng.random((self.size, self.n_feats)) > self.drop_p
        self.dropmask[self.traj_sp, :] = True   # trajectory starts are always valid
        self.dropmask[-1, :] = False


class PerFeatureDropWrapper(gym.Wrapper):
    """Env wrapper that drops each feature in drop_features independently with probability drop_p.
    info['feat_dropsteps'] contains per-feature dropstep counts for the returned observation.
    """
    def __init__(self, env, drop_p, drop_features, seed):
        super().__init__(env)
        self.drop_p = drop_p
        self.drop_features = list(drop_features)
        self.last_obs = None
        self.rng = np.random.default_rng(seed)
        self._feat_dropstep = np.zeros(len(drop_features), dtype=np.int32)

    def step(self, action):
        next_state, reward, done, trunc, info = self.env.step(action)
        obs = next_state.copy()
        for i, feat_idx in enumerate(self.drop_features):
            if self.rng.random() < self.drop_p:
                obs[..., feat_idx] = self.last_obs[..., feat_idx]
                self._feat_dropstep[i] += 1
            else:
                self.last_obs[..., feat_idx] = next_state[..., feat_idx]
                self._feat_dropstep[i] = 0
        info['feat_dropsteps'] = self._feat_dropstep.copy()
        return obs, reward, done, trunc, info

    def reset(self, seed):
        self.last_obs, info = self.env.reset(seed=seed)
        self._feat_dropstep[:] = 0
        return self.last_obs.copy(), info
