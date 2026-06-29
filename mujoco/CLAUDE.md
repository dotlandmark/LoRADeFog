# LoRADeFog / mujoco

MuJoCoの連続制御タスクにおいて、観測の一部が欠落（Drop）する状況でも堅牢に動作するDecision Transformerを、LoRAアダプタでファインチューニングするフレームワーク。

## ディレクトリ構成

```
mujoco/
├── main.py / main_mp.py   # エントリポイント（シングル / マルチプロセス）
├── core.py                # train() / eval() の実装
├── model.py               # DecisionTransformer
├── buffer.py              # SequenceBuffer（オフラインデータ管理とサンプリング）
├── drop_fn.py             # Dropping戦略クラス群
├── utils.py
├── cfgs/
│   ├── config.yaml        # メイン設定
│   └── env/
│       ├── walker2d.yaml  # Walker2d: rtg_target=5000, max_timestep=1000
│       └── hopper.yaml    # Hopper:   rtg_target=4000, max_timestep=1000
└── datasets/              # D4RL pkl形式データセット
```

## 実行コマンド

```bash
# シングルプロセス
python main.py

# 環境切り替え
python main.py env=hopper

# パラメータオーバーライド例
python main.py part_dropping.drop_p=0.3 part_dropping.drop_features=[0,1,2]

# GPU指定
CUDA_VISIBLE_DEVICES=1 python main.py
```

## Droppingの仕組み（3層アーキテクチャ）

### 1. バッファレベル（drop_fn.py）

`dropmask` と `dropstep` で各タイムステップ・各特徴量の有効性と「最後の有効フレームからの距離」を管理。

| クラス | dropmask/dropstep の形状 | 動作 |
|--------|--------------------------|------|
| `ConstFn` | `(buffer_size,)` | 全特徴量を同時に固定確率でDrop |
| `LinearFn` | `(buffer_size,)` | 全特徴量を同時に確率を線形増加させてDrop |
| `PerFeatureConstFn` | `(buffer_size, n_feats)` | 各特徴量を**独立に**固定確率でDrop |

`PerFeatureConstFn` は `part_dropping.use_part_dropping: true` のとき自動的に選択される。

### 2. モデルレベル（model.py）

`drop_aware: true` のとき、`embed_dropstep` でDropステップ距離を埋め込み、状態・リターン埋め込みに加算する。

- **グローバルDrop** (`dropsteps` shape: `(B, T)`): 1つの埋め込みを直接加算
- **特徴量ごとDrop** (`dropsteps` shape: `(B, T, n_feats)`): 特徴量ごとに埋め込んで**sum**してから加算

### 3. 訓練レベル（core.py）

1. **通常訓練**: `buffer.drop_cfg.drop_p` のDropでDecision Transformerを学習
2. **ファインチューニング**: `freeze_trunk()` で本体を固定し `finetune_drop_p`（デフォルト0.8）の高Drop率で適応

## config.yaml 主要設定

### part_dropping（特徴量ごとの独立Drop）

```yaml
part_dropping:
  use_part_dropping: true      # false にするとグローバルDrop（ConstFn）に戻る
  drop_features: [0, 1, 2]    # Dropする特徴量のインデックス（複数指定可）
  drop_p: 0.5                  # 各特徴量の独立Drop確率
```

`use_part_dropping: true` にすると `buffer.drop_cfg.drop_fn` が自動的に `per_feat_const` に切り替わる。`buffer.drop_cfg.drop_fn` を直接変更する必要はない。

### buffer.drop_cfg（グローバルDrop用）

```yaml
buffer:
  drop_cfg:
    drop_fn: const             # 'const' / 'linear' / 'per_feat_const'（自動設定）
    drop_p: 0.0                # 訓練時のDrop確率
    finetune_drop_p: 0.8       # ファインチューニング時のDrop確率
    update_interval: 100       # dropmaskを更新する訓練ステップ間隔
    drop_aware: true           # model.drop_aware と連動
```

### LoRA

```yaml
lora:
  use_lora: true
  r: 8
  lora_alpha: 16
  lora_dropout: 0.0
  lora_save_path: C:\vscodeC\DeFogLoRA\LoRADeFog\mujoco\model
```

LoRAの対象レイヤーは `q_net` と `v_net`（MaskedCausalAttentionのQ/Vプロジェクション）。

### 保存されるLoRAアダプタのフォルダ名

```
{lora_save_path}/{env_name}_{drop_p}_{feats_str}_seed_{seed}_{timestamp}/
```

例: `walker2d_0.5_feats0-1-2_seed_0_20260614_1234/`

per-feature Dropが無効のときは `featsnone`。

## buffer.sample() のDrop処理詳細

### グローバルDrop

```
observation_index = selected_index - dropsteps   # 全特徴量を同一インデックスから取得
rewards_to_go    = rtg[observation_index]        # rtgもシフト
```

### 特徴量ごとDrop（per-feature）

```
states[b, t, feat_idx] = self.states[selected_index - dropstep[b,t,i], feat_idx]  # 特徴量ごと個別参照
rewards_to_go          = rtg[selected_index]   # rtgはDropさせない（常に現在ステップ）
```

**RTGは欠落させない**。理由：元の研究ではRTGを欠落させる環境で実験していたが、今回の研究では特徴量ごとに欠落を行った結果を検証したいため、RTGは対象外だから。

## eval() のDrop対応

- `model.use_per_feat_dropping = True` のとき、`PerFeatureDropWrapper` を使用
- `info['feat_dropsteps']` から特徴量ごとのDropステップを取得し、モデルに `(1, T, n_feats)` 形状で注入
- RTGはシフトしない

## 特徴量別LoRAマージ（feature_lora）

### 概要

特徴量ごとに個別に学習されたLoRAアダプタを推論時にマージして適用する機構。現在DropされているFeatureに対応するアダプタだけを選択・合成する。

### Config

```yaml
feature_lora:
  use_feature_lora: false
  adapter_paths:              # アダプタ名（"feat_{state_index}"）→ 保存済みパス
    feat_0: /path/to/adapter_feat0
    feat_1: /path/to/adapter_feat1
    feat_2: /path/to/adapter_feat2
  merge_weight: 1.0           # 各アダプタの重み（1.0 = 各ΔW をそのまま加算）
  precompute_all: true        # 起動時に全 2^n 通りの組み合わせを事前マージ
```

- `adapter_paths` のキーはそのままPEFTのアダプタ名になる（`feat_{state_feature_index}` 形式）
- アダプタは `drop_features` に列挙した特徴量インデックスに対応して用意する
- `dynamic_lora` と `feature_lora` は同時使用不可

### 推論時の動作フロー

```
per_feat_dropstep を取得（各特徴量の経過ステップ数）
  ↓
active_set = {dropstep > 0 の特徴量インデックス}
  ↓
active_set が前ステップと同じ → スキップ（再マージなし）
  ↓
active_set == ∅ → disable_adapter_layers()（ベースモデルで推論）
  ↓
active_set に変化あり：
  キャッシュに "merged_{sorted_indices}" が存在？
    Yes → set_adapter()
    No  → add_weighted_adapter() でマージ → キャッシュに追加 → set_adapter()
```

### マージの仕組み

PEFTの `add_weighted_adapter(..., combination_type="linear")` を使用。

```
merged_ΔW = merge_weight × ΔW_feat_0 + merge_weight × ΔW_feat_2  （例：feat_0, feat_2 がDrop中）
```

`precompute_all: true` の場合は起動時に全組み合わせを作成済みのため、推論中に `add_weighted_adapter` は呼ばれない（高速）。特徴量数が 6 以下（64通り）なら有効推奨。

### アダプタ命名規則

| 用途 | 名前の例 |
| --- | --- |
| ソースアダプタ（feat 0 用） | `feat_0` |
| マージ済みアダプタ（feat 0, 2 がDrop） | `merged_0_2` |
| Drop なし（ベースモデル） | LoRA 無効化 |

## 動的LoRA（dynamic_lora）

評価時のDropRate（0.0〜0.9）に応じて10種のLoRAアダプタを動的切り替え。通常は `use_dynamic_lora: false`。

## 注意事項

- `model.drop_aware` と `buffer.drop_cfg.drop_aware` は必ず同じ値にする（config.yamlで `${model.drop_aware}` で参照しているため通常は自動）
- `load_model_path` に `.pt` 拡張子は不要
- データセットは `datasets/{env_name_lower}-{dataset}.pkl` 形式（例: `walker2d-medium.pkl`）
