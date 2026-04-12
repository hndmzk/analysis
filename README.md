# 市場解析・予測システム MVP

`market-prediction-agent-spec.md` を最上位要件として実装した、offline-safe 既定の市場解析・予測システムです。  
既定では `dummy` データで end-to-end 実行し、公開データを使う OOS 監査は別スクリプトで実行します。  
実売買は無効です。

## 主要機能

- データ取得
  - `dummy`
  - `polygon`
  - `alphavantage`
  - `stooq`
  - `fred_market_proxy`
  - `fred` / `fred_csv`
- モデル
  - LightGBM 3-class classifier
  - 時系列 calibration
  - Tree SHAP attribution
- 検証
  - walk-forward backtest
  - CPCV / fixed PBO proxy
- calibration metrics
- PSI drift
- feature stability audit
- HMM regime
- retraining monitor
- 出力
  - forecast / evidence / risk / report JSON
  - paper trading batch / ledger / weekly review / retraining event
  - monitor audit JSON

## セットアップ

### 推奨: `uv`

```powershell
uv sync
Copy-Item .env.example .env
```

### 代替: `pip`

```powershell
python -m pip install -e .[dev]
Copy-Item .env.example .env
```

## 実行コマンド

### dummy データ生成

```powershell
uv run python scripts/generate_dummy_data.py
uv run python scripts/generate_dummy_data.py --dummy-mode predictable_momentum
```

### full pipeline / backtest

```powershell
uv run python scripts/run_backtest.py
uv run python scripts/run_backtest.py --dummy-mode null_random_walk
uv run python scripts/run_backtest.py --dummy-mode predictable_momentum
uv run python scripts/run_backtest.py --dummy-mode predictable_momentum --comparison-models xgboost_multiclass_calibrated
```

### synthetic PBO sanity

```powershell
uv run python scripts/validate_pbo_ranges.py
```

### OOS monitor audit

synthetic:

```powershell
uv run python scripts/run_monitor_audit.py --dataset-type synthetic
```

public real-data:

```powershell
uv run python scripts/run_monitor_audit.py --dataset-type public_real_market
```

初回 live smoke 前に snapshot seed:

```powershell
uv run python scripts/seed_public_snapshots.py --tickers SPY,QQQ,DIA,GLD
```

seed してから監査まで 1 コマンドで行う場合:

```powershell
uv run python scripts/run_monitor_audit.py --dataset-type public_real_market --seed-public-snapshots
```

公開 upstream が不安定で seed に失敗した場合:

- 別のネットワーク環境で `scripts/seed_public_snapshots.py` を実行し、生成された `storage/public_data/snapshots/` をこの repo にコピーしてください
- その後に `scripts/run_monitor_audit.py --dataset-type public_real_market` を実行すると local snapshot fallback が効きます

両方:

```powershell
uv run python scripts/run_monitor_audit.py --dataset-type both
```

universe を変える場合:

```powershell
uv run python scripts/run_monitor_audit.py --dataset-type public_real_market --tickers SPY,QQQ,DIA,GLD
```

### schema 検証

```powershell
uv run python scripts/validate_schemas.py
```

既定では各 schema folder の最新 JSON を検証します。過去の全 artifact を再監査したい場合は `uv run python scripts/validate_schemas.py --all` を使ってください。

### test / lint / type check

```powershell
uv run pytest tests/ -v
uv run ruff check src/ tests/
uv run mypy src/
```

## synthetic と public real-data audit の違い

### synthetic audit

- `scripts/validate_pbo_ranges.py`
- `scripts/run_monitor_audit.py --dataset-type synthetic`
- deterministic な regression / sanity 用です
- 市場証拠や投資判断には使いません

### public real-data audit

- Source priority is `yahoo_chart -> fred_market_proxy`, and macro stays `fred_csv`.
- `scripts/seed_public_snapshots.py` は初回 live smoke 前の seed 手順です。cache と `storage/public_data/snapshots/` を温め、後続の public audit で local fallback を使えるようにします。
- `monitor_audit.data_sources.proxy_ohlcv_used=true` は `fred_market_proxy` を使ったことを明示します。`fred_market_proxy` は public close series から OHLCV / volume proxy を構成する research-only adapter です。

- 単発監査:

```powershell
uv run python scripts/run_monitor_audit.py --dataset-type public_real_market --tickers SPY,QQQ,DIA,GLD --as-of-date 2026-04-03
```

- OHLCV の優先順は `yahoo_chart -> fred_market_proxy`
- macro は `fred_csv`
- `fred_market_proxy` は public close series を OHLCV / volume proxy に変換します
- `stooq` adapter も実装していますが、監査の既定経路ではありません
- 公開データ取得は `retry -> fresh cache -> compatible cache -> stale cache -> compatible stale cache -> local snapshot fallback` の順で復旧します
- cache / snapshot / retry は `config/default.yaml` の `data.public_data.*` で制御します
- `compatible cache` / `compatible snapshot` は、より広い date range の既存 payload が要求範囲を完全に包含していれば再利用します。複数日の `--as-of-date` 監査を繰り返すときの初回失敗率を下げるための経路です。
- local snapshot fallback は、過去の成功取得で `storage/public_data/snapshots/` に保存済みの payload があるときに使われます
- `monitor_audit.data_sources.proxy_ohlcv_used=true` のとき、その監査は proxy OHLCV を使っています
- drift 監査は raw PSI に加えて stability-adjusted PSI、feature family、proxy-sensitive 判定、supplementary analysis を出します
- supplementary analysis は「proxy artifact 寄り」か「true regime shift 寄り」かの切り分け補助です
- upstream 更新や可用性により結果は変動します
- research-only です
- 複数日・複数 universe の repeated OOS 監査:

```powershell
uv run python scripts/run_public_audit_suite.py --as-of-dates 2026-03-20,2026-03-27,2026-04-03 --ticker-sets "SPY,QQQ,DIA,GLD|SPY,QQQ,GLD" --history-days 1100 --cpcv-max-splits 2
```

- suite 出力は `information_ratio`, `pbo`, `selection_stability`, `retraining_rate` の分布を日付別・ticker set 別に要約します。
- 12 dates の拡張検証:

```powershell
uv run python scripts/run_public_audit_suite.py --as-of-dates 2026-01-16,2026-01-23,2026-01-30,2026-02-06,2026-02-13,2026-02-20,2026-02-27,2026-03-06,2026-03-13,2026-03-20,2026-03-27,2026-04-03 --ticker-sets "SPY,QQQ,DIA,GLD|SPY,QQQ,GLD" --history-days 1100 --cpcv-max-splits 2
```

- 最新 12-date suite では `base_retraining_rate=0.9583` に対して `retraining_rate=0.4167` まで低下し、同時に `IR mean=3.5166`, `PBO mean=0.1875`, `selection_stability mean=0.9005` を維持しました。

### drift / feature stability

- `backtest_result.drift_monitor.family_thresholds` stores per-family warning / critical PSI thresholds.
- `config/default.yaml` exposes `risk.drift.proxy_sensitive_features` so proxy-sensitive features are configured explicitly instead of hard-coded.
- `backtest_result.drift_monitor.supplementary_analysis.feature_diagnostics[]` contains per-feature `raw_psi`, `adjusted_psi`, `family`, `proxy_sensitive`, `primary_cause`, and `retrain_action`.

- volatility / macro / volume 系は drift 監査時に signed-log と winsorization をかけた stability-adjusted PSI も計算します
- `backtest_result.drift_monitor.max_psi` は adjusted PSI です
- `backtest_result.drift_monitor.max_raw_psi` に raw PSI を残します
- `backtest_result.drift_monitor.supplementary_analysis.primary_cause` は主因推定です
  - `proxy_artifact_likely`
  - `regime_shift_likely`
  - `mixed_proxy_and_regime`
  - `mixed_or_unknown`
  - `stable`
- public audit で `proxy_ohlcv_used=true` かつ drift が proxy-sensitive features に偏る場合、retraining trigger は自動発火せず watch 扱いに落とします
- regime shift / transition / non-proxy drift が重なる場合は retraining trigger を維持します

## retraining monitor

- Drift gate は raw feature count ではなく family-weighted score を使います。`risk.drift.family_trigger_weights` で macro / volatility を重く、price_momentum / volume を軽く扱います。
- 通常の drift trigger は persistence 条件付きです。`weighted_score >= model.retraining.drift_family_weight_threshold` に加えて、`drift_min_persistent_features` と `drift_min_persistent_families` を満たした場合だけ `feature_drift` を trigger します。
- 例外として `weighted_score >= model.retraining.drift_immediate_weight_threshold` のときだけ即時 trigger を許可します。
- Calibration gate も fold persistence を使います。`ece` / `calibration_gap` の warning は、`calibration_min_persistent_folds` 以上の fold かつ `calibration_min_fold_breach_ratio` 以上で継続しない限り trigger しません。平均 breach が大きい場合のみ `calibration_immediate_multiplier` による即時 trigger を許可します。
- Regime-conditioned threshold を入れています。high-vol / transition では drift と calibration の判定感度を変え、`family_retrain_suppression` で price-momentum / volume family を regime 別に watch 扱いへ落とせます。
- Run-to-run cool-off を入れています。直近の effective retrain から `cooloff_business_days` 以内で、同じ regime bucket の繰り返し trigger なら `should_retrain=false` に落とします。既定の override は新規 `regime_shift` のみです。
- `retraining_monitor` には `base_should_retrain`, `policy_decision`, `policy_notes`, `cooloff_active` を残します。suite では `base_retraining_rate` と `retraining_rate` を並べて、policy の抑制効果を比較します。
- `retraining_monitor.drift_signal` と `retraining_monitor.calibration_signal` に weighted score、persistent feature/family count、fold breach ratio を残します。
- `monitor_audit.notes[]` に drift gate と calibration gate の要約が入り、単発ノイズか継続異常かを追跡できます。
- 2026-04-03 の 12-date public OOS suite では effective trigger は `feature_drift=6`, `regime_shift=6`, `pbo=2` でした。regime shift がない run の多くは cool-off で suppress され、単発ノイズ由来の再学習が減ります。

## XGBoost 比較

- 既定の primary model は `lightgbm_multiclass_calibrated` です
- XGBoost 比較は opt-in で、`--comparison-models xgboost_multiclass_calibrated` または `model.comparison_models` で有効化します
- 比較結果は `backtest_result.model_comparison[]` に出力します
- 比較結果は benchmark 用であり、approval の主判定は primary model 側を基準にします

## dummy モード

### `null_random_walk`

- sanity check 用
- ほぼ予測不能
- 3-class hit rate が概ね `0.33` 近辺に収まることを期待します
- `AUTO_APPROVED` にはなりません

### `predictable_momentum`

- real-like synthetic / 開発用
- momentum / regime 構造を持つ synthetic データです
- 実市場データの代替ではありません

## PBO の固定定義

- CPCV の各 split で in-sample 最良 candidate を選ぶ
- その candidate が out-of-sample で中央値未満に落ちた split の比率を PBO とする

解釈:

- `< 0.20`: `low_overfit_risk`
- `0.20 - 0.49`: `moderate_overfit_risk`
- `0.50 - 0.79`: `high_overfit_risk`
- `>= 0.80`: `severe_overfit_risk`

## public real-data monetization diagnostics

- The default portfolio rule remains `classified_directional`. `probability_threshold`, `top_bucket_fraction`, `bottom_bucket_fraction`, and `holding_days` stay configurable, but CPCV best candidates are kept as parallel comparisons and are not auto-promoted into the default rule.
- `backtest_result.portfolio_rule_analysis` compares that selected rule against the legacy strict two-sided rule. This is the main place to explain cases where hit rate exists but realized IR is weak.
- `monitor_audit.backtest.portfolio_rule_analysis` and `monitor_audit.notes[]` summarize:
  - top / bottom bucket settings
  - holding rule
  - participation-aware rebalance thinning via `max_turnover_per_day` and liquidity proxy scaling
  - `min_edge` filter, `bucket_hysteresis`, `hysteresis_edge_buffer`, and `reentry_cooldown_days`
  - turnover and cost drag contribution
  - selection stability
  - controlled vs uncontrolled selected-rule comparison through `portfolio_rule_analysis.control_effect`
  - primary reasons why monetization improved or failed
- The current default OOS configuration uses `min_edge=0.02`, `hysteresis_edge_buffer=0.01`, `reentry_cooldown_days=2`, and `max_turnover_per_day=0.06`. These controls were tuned against public OOS diagnostics, not by auto-promoting the CPCV best candidate.
- CPCV now evaluates a compact candidate family across `strategy_name`, `probability_threshold`, `top_bucket_fraction`, `bottom_bucket_fraction`, and `holding_days`, then reports grouped summaries for threshold, bucket, holding-rule, and strategy sensitivity.
- For small public universes, the legacy `classified_two_sided` rule can leave most days flat because one-sided signals dominate. The audit keeps that legacy comparison explicitly so the failure mode is visible instead of hidden.
- Shared handoff artifact: use `dist/package_clean.zip`.

## approval

`risk_review.approval` は [builders.py](src/market_prediction_agent/reporting/builders.py) で判定します。

- `AUTO_APPROVED`
  - `FAIL` がない
  - `WARNING` がない
  - drawdown 条件に抵触しない
- `MANUAL_REVIEW_REQUIRED`
  - `WARNING` が 1 つ以上ある
  - drawdown または warning 数が高リスク域
  - `null_random_walk` の sanity run
- `BLOCKED`
  - `FAIL` が 1 つ以上ある

`approval_notes` に根拠を残します。  
dummy run と public real-data audit は approval にかかわらず research-only として扱ってください。

## paper trading

paper trading は実売買ではなく、forecast を ledger に記録し、後続 run で擬似的に損益確定します。

実装している制約:

- round-trip fee
- round-trip slippage
- signal time と execution time の分離
  - signal は forecast 生成時刻
  - execution は `trading.execution_delay_business_days` 後の `trading.execution_time_utc`
- gap-open execution
  - signal close から execution open までの adverse move を `gap_slippage_bps` として別記録
- intended / filled / unfilled の分離
  - `target_notional` / `intended_order_quantity` は signal 時点
  - `executed_notional` / `filled_quantity` / `unfilled_quantity` は execution 時点
- liquidity cap
  - `trading.max_participation_rate`
  - `trading.min_daily_dollar_volume`
  - `trading.min_trade_notional`
  - `trading.adv_lookback_days` の daily ADV から fillable size を計算
- partial fill
  - ADV 連動 participation cap を超える分は `unfilled` として ledger に残す
- `FLAT` は `SKIPPED`

execution diagnostics:

- `fill_rate`: `sum(filled_quantity) / sum(intended_order_quantity)`
- `partial_fill_rate`: partial fill になった execution 件数の比率
- `missed_trade_rate`: intended order が 0 fill で終わった件数の比率
- `realized_vs_intended_exposure`: `sum(executed_notional) / sum(target_notional)`
- `execution_cost_drag`: round-trip cost と gap-open drag を合わせた execution 由来の drag
- `gap_slippage_bps`: signal close 대비 execution open の adverse move

主な execution diagnostics の出力先:

- `outputs/paper_trading/<date>/...json`
- `outputs/paper_trading/trade_ledger.parquet`
- `outputs/weekly_reviews/<week_id>/...json`
- monitor audit の `paper_trading_summary`

主な出力:

- `outputs/paper_trading/<date>/...json`
- `outputs/paper_trading/trade_ledger.parquet`
- `outputs/weekly_reviews/<week_id>/...json`
- `outputs/retraining_events/<date>/...json`

## 設定メモ

- `risk.drift.bucket_count`
- `risk.drift.price_momentum.*`
- `risk.drift.volatility.*`
- `risk.drift.volume.*`
- `risk.drift.macro.*`
- `risk.drift.calendar.*`
- `risk.drift.proxy_sensitive_features`

主な追加項目:

- `data.primary_source`
- `data.fallback_source`
- `data.macro_source`
- `data.default_tickers`
- `data.public_data.cache_path`
- `data.public_data.snapshot_path`
- `data.public_data.cache_ttl_hours`
- `data.public_data.retry_count`
- `data.public_data.retry_backoff_seconds`
- `trading.paper_capital`
- `trading.min_trade_notional`
- `trading.min_daily_dollar_volume`
- `trading.max_participation_rate`

## clean export

clean copy:

```powershell
uv run python scripts/package_clean.py
```

zip まで作る:

```powershell
uv run python scripts/package_clean.py --zip
```

共有用成果物は `dist/package_clean.zip` を使ってください。`--zip` は `dist/package_clean/` を作ったうえで、その clean copy から zip を生成します。  
zip には `.claude/`, `.venv/`, `__pycache__/`, `*.egg-info/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.pytest_tmp/`, `pytest-cache-files-*`, repo ルートの `storage/` を含めません。`src/market_prediction_agent/storage/` のコードは含まれます。

## 制約

- 実売買は未実装です
- `trading.enable_live_trading` は常に `false` です
- external API を使う test は追加していません。network が必要な箇所は test で mock します
- public real-data audit は公開データ源の可用性に依存します
- `fred_market_proxy` の intraday range と volume は proxy 値です
- runtime で作る cache / snapshot は clean export に含めません
- PBO は portfolio-rule family に対する proxy です

## 未実装事項

- XGBoost
- 広い universe の public audit
- SEC / ニュース / ファンダメンタル / セクター特徴量
- paper trading の板モデル
- live API の結合試験
- Pydantic 設定

## 対応表

詳細は [spec_mapping.md](docs/spec_mapping.md) を参照してください。

## retraining policy update

- `outputs/retraining_events/retraining_event_ledger.parquet` を stateful retraining policy の履歴元に使います。
- PBO は `model.retraining.pbo_min_persistent_observations`, `pbo_persistence_business_days`, `pbo_immediate_threshold` で persistence / immediate trigger を分けます。単発の PBO warning だけでは即 retrain しません。
- drift / calibration / PBO は family × regime cause key で履歴化し、同一原因の再発は `cause_cooloff_business_days` の間 suppress します。
- `regime_shift` 単独は `state_probability` と `transition_rate` が benign なら watch-only に落とします。`transition_regime` や non-proxy drift が併発したときだけ retrain 側へ戻ります。
- live single-run では event ledger を自動読込し、public suite では同じ row 形式の history を replay して policy を検証します。
- 2026-04-03 の 12-date public OOS suite は policy replay で再評価しました。backtest 自体は不変なので `IR` / `PBO` / `selection_stability` 分布は維持され、effective `retraining_rate` は `0.4167 -> 0.2500` まで低下しました。
- replay artifact: `storage/outputs/monitor_audit_suites/public_real_market/2026-04-03/1841cc91-fad7-4e7d-bfe9-7240872cf5fe.json`

## retraining replay and audit profiles

- `src/market_prediction_agent/retraining/ledger_service.py` is the dedicated service for `outputs/retraining_events/retraining_event_ledger.parquet`. Live runs and paper-trading updates now use the same storage path and the same history normalization.
- `scripts/run_public_audit_suite.py` supports `--profile fast|standard|full_light|full`.
  - `fast`: 3 weekly as-of dates, 1 ticker set, CPCV split cap 1
  - `standard`: 6 weekly as-of dates, 2 ticker sets, CPCV split cap 2
  - `full_light`: 12 weekly as-of dates, 1 ticker set, CPCV split cap 1
  - `full`: 12 weekly as-of dates, 2 ticker sets, CPCV split cap 2
- The default history length is intentionally kept on the safe side for all profiles because public OOS runs need enough warm-up to build at least one valid walk-forward window.
- The retraining policy is now tuned more conservatively for operations:
  - `drift_family_weight_threshold=2.45`
  - `drift_immediate_weight_threshold=2.95`
  - `pbo_warning=0.60`
  - `pbo_min_persistent_observations=3`
  - `pbo_persistence_business_days=45`
  - `pbo_immediate_threshold=0.85`
- Calibration governance now needs both fold persistence and run persistence:
  - `calibration_min_persistent_folds=2`
  - `calibration_min_fold_breach_ratio=0.40`
  - `calibration_min_persistent_runs=2`
  - `calibration_min_persistent_span_business_days=10`
  - `calibration_persistence_business_days=30`
- On the 2026-04-03 stored 12-date full suite replay, these changes moved `base_retraining_rate` from `0.9583` to `0.6667` and effective `retraining_rate` from `0.4167` to `0.1667` while leaving IR / PBO / selection stability unchanged because replay reuses the same backtests.
- On the latest fresh live `full_light` suite, calibration governance reduced `base_retraining_rate` to `0.3333` while overall effective `retraining_rate` stayed `0.0833` because the remaining effective run was drift-driven, not calibration-driven.
- Drift governance now adds a targeted stable-transition suppression for `volume` family noise. This reduced `base_retraining_rate` again to `0.2500`, but the remaining effective run still stayed at `0.0833` because it was co-triggered by `transition_regime`.
- `scripts/run_retraining_policy_replay.py` is the formal replay entrypoint. It reuses stored backtests and replays only the retraining policy, so IR / PBO / selection stability stay fixed while `base_retraining_rate` and effective `retraining_rate` are re-evaluated under the current policy.
- `base_retraining_rate` means raw trigger rate before cool-off / suppression. `retraining_rate` means the final effective trigger rate after persistence, family suppression, and stateful cool-off.

Example commands:

```powershell
uv run python scripts/run_public_audit_suite.py --profile fast --seed-public-snapshots
uv run python scripts/run_public_audit_suite.py --profile full_light --seed-public-snapshots
uv run python scripts/run_public_audit_suite.py --profile full
uv run python scripts/run_retraining_policy_replay.py --suite-path storage/outputs/monitor_audit_suites/public_real_market/2026-04-03/9d33c0da-148e-4298-894b-1e7e7361fe5e.json --profile full
```

## pbo diagnostics and suite roles

- `cpcv.pbo_diagnostics` now decomposes raw PBO by candidate family, threshold, bucket pair, and `holding_days`.
- `cpcv.cluster_adjusted_pbo`, `cluster_adjusted_pbo_summary`, and `cluster_adjusted_pbo_diagnostics` now report the same CPCV audit after parameter-cluster regrouping.
- `cpcv.pbo_diagnostics.near_candidate_competition` summarizes close winner-vs-runner-up contests with in-sample margin, OOS margin, and a `competition_dominated` flag.
- Parameter clusters are defined by `model.cpcv.threshold_cluster_tolerance`, `bucket_cluster_tolerance`, and `holding_days_cluster_tolerance`.
- Operationally, `full_light` is the routine live audit profile. `full` is reserved for replay and deeper investigation of stored artifacts.
- The PBO retraining gate now needs both repeated runs and enough elapsed business-day span:
  - `pbo_min_persistent_observations`
  - `pbo_min_persistent_span_business_days`
  - `pbo_persistence_business_days`
- Severe standalone PBO is downgraded to watch-only when CPCV shows close-candidate competition dominating the grid instead of a broad parameter-family breakdown.
- `monitor_audit_suite` now carries both `pbo` and `cluster_adjusted_pbo`, plus both candidate-level and adjusted competition-dominance rates.
- `monitor_audit_suite.distribution_summary.calibration_dominated_analysis` breaks effective calibration-driven runs down by fold persistence, run persistence, regime, ticker set, and transport source.
- `monitor_audit_suite.distribution_summary.drift_dominated_analysis` breaks raw/effective drift runs down by family, regime, proxy-sensitive mix, transport source, and history span, and reports which counterfactual lever would help:
  - `family_persistence_relief_count`
  - `severity_threshold_low_vol_relief_count`
  - `regime_suppression_relief_count`
  - plus `*_effective_relief_count` fields to show whether that lever would actually reduce effective retraining, not just raw drift alerts.
- `monitor_audit_suite.distribution_summary.regime_dominated_analysis` breaks regime-driven runs down by transition persistence span, `state_probability` bucket, `transition_rate` bucket, and co-trigger drift family.
- Regime governance now separates `stable_transition`, `watch_transition`, and `unstable_transition`.
  - `stable_transition` is allowed to use transition-specific family suppression and stays watch-only.
  - `transition_regime` only remains effective when the transition is genuinely unstable after persistence checks.
  - `regime_shift` is kept as watch-only unless feature drift co-occurs in the same run.
- In the latest live `full_light` run, cluster-adjusted PBO reduced competition-dominated runs to `0.0`, and calibration governance reduced calibration-dominated effective runs to `0`.
- Latest 2026-04-03 full replay artifact:
  - `base_retraining_rate=0.6667`
  - `retraining_rate=0.1667`
  - artifact: `storage/outputs/monitor_audit_suites/public_real_market/2026-04-03/565d5e54-23f9-49aa-8be4-0a736ca7a085.json`
- Latest 2026-04-04 fresh live `full_light` artifact:
  - `base_retraining_rate=0.1667`
  - `retraining_rate=0.0000`
  - candidate-level `pbo mean=0.4167`
  - cluster-adjusted `pbo mean=0.0000`
  - `candidate_pbo_competition_dominated_rate=0.4167`
  - `pbo_competition_dominated_rate=0.0000`
  - `calibration_dominated_analysis.run_count=0`
  - `drift_dominated_analysis.raw_run_count=2`
  - `drift_dominated_analysis.effective_run_count=0`
  - `drift_dominated_analysis.regime_suppression_relief_count=2`
  - `regime_dominated_analysis.raw_run_count=2`
  - `regime_dominated_analysis.effective_run_count=0`
  - artifact: `storage/outputs/monitor_audit_suites/public_real_market/2026-04-04/d90a30d9-487b-4d55-b61d-5b51c7a44c81.json`

## audit report rendering

- `scripts/render_audit_report.py` renders persisted artifacts into:
  - machine-readable `audit_report` JSON
  - review-friendly Markdown with fixed headings
- The renderer does not recompute model logic. It only reshapes persisted artifacts from:
  - `monitor_audit`
  - `monitor_audit_suite`
  - `weekly_review`
  - `backtest_result`
  - `paper_trading_batch`
- Heading order is fixed:
  1. Executive Summary
  2. Run Context
  3. Model / Portfolio Performance
  4. Execution Diagnostics
  5. Drift / Calibration / Regime Review
  6. Retraining Decision
  7. Watch-Only Findings
  8. Data Sources and Transport
  9. Top Risks
  10. Recommended Next Actions
  11. Artifact References
- `full_light` is the routine live monitoring profile. `full` is the replay / deep-dive profile.
- raw signal and effective trigger are rendered separately, and candidate-level PBO is shown next to cluster-adjusted PBO.
- Default output path:
  - `storage/outputs/audit_reports/<report_type>/<YYYY-MM-DD>/<report_id>.json`
  - `storage/outputs/audit_reports/<report_type>/<YYYY-MM-DD>/<report_id>.md`

Examples:

```powershell
uv run python scripts/render_audit_report.py --latest-monitor-audit
uv run python scripts/render_audit_report.py --latest-suite --format markdown
uv run python scripts/render_audit_report.py --input storage/outputs/monitor_audits/synthetic/2026-04-04/364650f5-0191-4d17-86f2-d4b94cdc109b.json --format json --output storage/outputs/audit_reports/manual/report.json
```

## feature expansion

- Priority order is news, fundamentals, then sector features.
- Added feature families:
  - news: `news_sentiment_1d`, `news_sentiment_5d`, `news_sentiment_decay_5d`, `news_relevance_5d`, `news_novelty_5d`, `news_source_diversity_5d`, `news_volume_zscore_20d`
  - fundamentals: `fundamental_revenue_growth`, `fundamental_earnings_yield`, `fundamental_leverage`, `fundamental_profitability`
  - sector: `sector_relative_momentum_20d`, `sector_strength_20d`, `sector_vol_spread_20d`
- News now uses a live adapter by default and keeps the proxy as fallback:
  - primary news source: `yahoo_finance_rss`
  - fallback news source: `offline_news_proxy`
  - news transport metadata is recorded under `monitor_audit.data_sources.feature_sources.news.transport`
  - news lineage keeps `requested_source`, `used_source`, `fallback_used`, `transport`, `missing_rate`, and `stale_rate`
  - live news feature engineering now improves `headline -> ticker` mapping with symbol + company-name/alias matching and adds neutral no-news rows so coverage is not lost purely because a day had no headlines
  - asset-level news aggregation now compares fixed lookback and decay-weighted variants, and exposes `headline_count`, sentiment, novelty, and source-diversity utility comparisons
- Fundamentals now use a live adapter by default and keep the proxy as fallback:
  - primary fundamentals source: `sec_companyfacts`
  - fallback fundamentals source: `offline_fundamental_proxy`
  - fundamentals transport metadata is recorded under `monitor_audit.data_sources.feature_sources.fundamental.transport`
- Other current sources remain explicit in artifacts:
  - sector: `static_sector_map`
- `backtest_result.feature_catalog[]` records `feature_family`, `data_source`, `missing_rate`, and `stale_rate` for every feature.
- `backtest_result.feature_importance_summary[]` and `feature_family_importance_summary[]` keep SHAP contribution tracking aligned with the same lineage metadata.
- `monitor_audit.feature_lineage` and `monitor_audit.data_sources.feature_sources` expose the same lineage for review.
- `audit_report` keeps the same section structure and shows the news/fundamental `used_source`, `requested_sources`, fallback flag, transport metadata, missingness, staleness, and news utility comparison without changing the existing headings.
- `audit_report` surfaces feature family contribution under `Model / Portfolio Performance` and feature source / missingness / staleness summaries under `Data Sources and Transport`.
- live smoke helpers:
  - `uv run python scripts/run_news_live_smoke.py --tickers SPY,QQQ,GLD --lookback-days 7`
  - `uv run python scripts/run_fundamentals_live_smoke.py --tickers AAPL,MSFT --lookback-days 1300`
- news live ingestion is now session-aware and multi-source. `yahoo_finance_rss` remains the primary live source, `google_news_rss` is the secondary live source, and `offline_news_proxy` remains the offline fallback.
- `google_news_rss` now uses multiple query variants per ticker (`aliases_or`, `primary_alias`, `ticker_stock`) so the secondary source is less brittle and source diversity is available in live runs instead of only in proxy fallback scenarios.
- session alignment is applied before asset-level aggregation:
  - `pre_market`: same-session signal date
  - `regular`: next business-day signal date
  - `post_market`: next business-day signal date
  - `weekend_shifted`: next business-day signal date
- news utility evaluation is now aligned to the signal day that results from session-aware normalization. `abs_ic` and `overlap_rate` are measured against signal-day close-to-close returns rather than a fixed next-day label, so pre-market / regular / post-market headlines are compared on a consistent calendar basis.
- `monitor_audit_suite.distribution_summary.news_utility_comparison` compares news coverage and utility by ticker set, transport origin, lookback window, decay variant, feature variant, source-weighted/session-weighted variant, session bucket, source mix, source-session bucket, and source-diversity bucket.
- the same suite summary now also breaks utility out by used source and source mix, so live runs can be compared across:
  - ticker set
  - transport origin
  - used source
  - source mix
  - lookback / decay variant
  - session bucket
- `audit_report` for single and suite artifacts mirrors the same news utility comparison under `Data Sources and Transport > feature_sources > news`, including `coverage`, `abs_ic`, `overlap_rate`, `source_diversity`, `weighted_variants`, `source_advantage_analysis`, and `session_bucket` comparisons.
- utility optimization is now measured against `abs_ic` and `overlap_rate`, not just headline coverage. The production daily news score remains source+session weighted, while unweighted / source-weighted / session-weighted / source+session-weighted variants are reported side by side for audit.
- the suite summary now carries `news_feature_coverage`, `news_feature_missing_rate`, `news_feature_staleness`, `news_used_source_counts`, and `news_transport_origin_counts` side by side, so coverage gains can be checked against missingness and transport provenance instead of read in isolation.
- in the latest live equity `full_light` suite, news utility no longer improves only on coverage. The summary shows:
  - baseline 1d `abs_ic` mean `0.1707`
  - best aggregation `10d` `abs_ic` mean `0.1953`
  - baseline overlap rate mean `0.9048`
  - best source mix `google_news_rss` overlap rate mean `0.9849`
  - best session bucket `mixed` `abs_ic` mean `0.2384`
- Policy defaults, retraining governance, and report section order are unchanged in this phase.
