# 市場解析・予測エージェント基盤 — 実装仕様書

> **Version**: 1.0.0  
> **Date**: 2026-04-02  
> **Author**: Research Architect (Claude)  
> **Target**: Implementation Agent (Codex)  
> **Status**: Implementation-Ready Draft

---

## 1. エグゼクティブサマリー

### システムの狙い

マルチエージェント構成による市場解析・予測基盤を構築する。目的は「根拠付きの市場見通し生成」であり、自動売買ではない。予測結果は構造化 JSON として出力し、人間が最終判断を行う。

### 対象市場

| 区分 | 対象 | 優先度 |
|------|------|--------|
| 米国株式 | S&P 500 構成銘柄（～500 銘柄） | **P0** |
| 暗号資産 | BTC, ETH, 時価総額上位 20 | **P1** |
| 日本株式 | TOPIX Core30 | **P2** (将来拡張) |

### 予測対象

- **主**: 日次リターンの方向分類（上昇 / 下落 / 横ばい）
- **副**: 5 日先期待リターン回帰、実現ボラティリティ予測（20 日窓）
- **補助**: イベント反応予測（決算発表後 1-3 日の異常リターン）

### 推奨アーキテクチャ

5 エージェント構成（後述）。各エージェントは独立プロセスで動作し、メッセージキュー（ローカルでは Redis Streams、将来は Cloud Pub/Sub）で非同期通信する。LLM 呼び出しは Research Agent と Report Agent に限定し、Forecast Agent は純粋な ML パイプラインとする。

### 実装優先順位

1. **Phase 0** (Week 1-2): データ取得パイプライン + ストレージ
2. **Phase 1** (Week 3-4): 特徴量生成 + ベースラインモデル + walk-forward 評価
3. **Phase 2** (Week 5-6): Research Agent (LLM + ニュースセンチメント)
4. **Phase 3** (Week 7-8): Risk/Critic Agent + Report Agent + 統合テスト
5. **Phase 4** (Week 9-10): ペーパートレード + ドリフト監視 + 運用化

---

## 2. 要件定義

### 2.1 対象資産クラス

| 資産クラス | ティッカー例 | データソース | 最小粒度 |
|------------|-------------|-------------|----------|
| 米国個別株 | AAPL, MSFT, NVDA | Polygon.io / Alpha Vantage | 1 分足 |
| 米国 ETF | SPY, QQQ, IWM | 同上 | 1 分足 |
| 暗号資産 | BTC-USD, ETH-USD | CoinGecko / Binance | 1 分足 |
| 米国債 | ^TNX (10Y Yield) | FRED | 日次 |
| コモディティ | GC=F (Gold), CL=F (Oil) | Alpha Vantage | 日次 |

### 2.2 予測ホライズン

| ホライズン | 用途 | ターゲット変数 |
|-----------|------|---------------|
| T+1 日 | 方向分類 | sign(close_t+1 / close_t - 1) 3 クラス |
| T+5 日 | リターン回帰 | (close_t+5 / close_t - 1) |
| T+20 日 | ボラティリティ | realized_vol_20d |
| T+1〜3 日 (イベント) | 決算反応 | abnormal_return vs sector |

**3 クラス定義**:
- UP: return > +0.5%
- DOWN: return < -0.5%
- FLAT: -0.5% ≤ return ≤ +0.5%

閾値は学習データの分布に応じて調整可能にする（設定ファイルで管理）。

### 2.3 成功指標

| 指標 | 方向分類 | リターン回帰 | ボラティリティ |
|------|---------|-------------|--------------|
| 主指標 | Hit Rate > 0.40 (3クラス, ランダム=0.33) | IC (情報係数) > 0.03 | MAE < naive baseline |
| 副指標 | Log Loss < 1.05 | Rank IC > 0.03 | MAPE < 30% |
| 校正 | Brier Score | — | — |
| 実運用 | Information Ratio > 0.3 (仮想 L/S) | — | — |
| リスク | Max Drawdown < 15% (仮想 L/S) | — | — |

**注意**: Hit Rate 0.40 は 3 クラス分類でのランダム (0.33) を 21% 上回る水準であり、取引コスト控除前の数値。実運用での有意性は Information Ratio で判断する。過度な精度期待は禁物。文献上、日次方向予測の安定的な精度は 52-58% (2クラス) が現実的な上限とされる [López de Prado 2018, "Advances in Financial Machine Learning", Wiley]。

### 2.4 非機能要件

| 要件 | 仕様 |
|------|------|
| 再現性 | 全パイプラインを seed 固定 + データバージョニング (DVC or git-lfs) で再現可能にする |
| レイテンシ | 日次バッチ: 全銘柄予測を 30 分以内。リアルタイムは Phase 4 以降 |
| コスト | API 月額 $0 (無料枠) ～ $200 (有料枠)。LLM コストは月 $50 以内を目安 |
| 監査可能性 | 全予測に evidence_bundle (入力特徴量 + モデル版 + タイムスタンプ) を紐付け |
| 失敗時の挙動 | データ取得失敗 → stale data フラグ + 前回値使用 + アラート。モデル推論失敗 → 予測なし (NaN) + アラート。決して「推測」で穴埋めしない |

---

## 3. データ設計

### 3.1 必須データソース一覧

| # | カテゴリ | ソース | API | 無料枠 | 更新頻度 | 用途 |
|---|---------|--------|-----|--------|---------|------|
| D1 | 株価 OHLCV | Polygon.io | REST | 5 req/min, EOD | 日次 / 分足 | 特徴量の基盤 |
| D2 | 株価 OHLCV (バックアップ) | Alpha Vantage | REST + MCP | 25 req/day (free) | 日次 / 分足 | D1 障害時のフォールバック |
| D3 | 暗号資産 OHLCV | CoinGecko | REST | 30 req/min | 日次 / 分足 | 暗号資産特徴量 |
| D4 | 暗号資産リアルタイム | Binance | REST + WS | Weight制 1200/min | リアルタイム | 暗号資産ストリーム (Phase 4) |
| D5 | 企業開示 (10-K/10-Q/8-K) | SEC EDGAR | REST (認証不要) | 制限なし (10 req/sec) | 日次チェック | ファンダメンタル特徴量 |
| D6 | 企業財務 XBRL | SEC EDGAR data.sec.gov | REST | 制限なし | 四半期 | 財務比率計算 |
| D7 | マクロ経済指標 | FRED | REST | 制限なし (API key 必要) | 日次～月次 | マクロ特徴量 |
| D8 | ニュース | Finnhub / NewsAPI | REST | 60 req/min / 100 req/day | 随時 | センチメント分析入力 |
| D9 | テクニカル指標 | 自前計算 (TA-Lib / pandas-ta) | — | — | 日次 | テクニカル特徴量 |

### 3.2 保存形式・キー設計

```
storage/
├── raw/                    # 取得したままの JSON/CSV
│   ├── ohlcv/{source}/{ticker}/{date}.parquet
│   ├── filings/{cik}/{accession}.json
│   ├── macro/{series_id}.parquet
│   └── news/{date}/{source}_{hash}.json
├── processed/              # 正規化済み
│   ├── features/{ticker}/{date}.parquet
│   └── labels/{ticker}/{date}.parquet
├── models/                 # 学習済みモデル
│   └── {model_name}/{version}/
└── outputs/                # 予測結果
    └── forecasts/{date}/{run_id}.json
```

**主キー構成**:
- OHLCV: `(ticker, timestamp_utc)`
- Filings: `(cik, accession_number, filing_date)`
- Macro: `(series_id, date)`
- News: `(source, article_id, published_at_utc)`
- Features: `(ticker, as_of_date, feature_version)`
- Forecasts: `(ticker, forecast_date, model_version, run_id)`

### 3.3 タイムゾーン

- **内部統一**: UTC
- **米国株イベント**: US/Eastern で受信 → UTC 変換して保存
- **暗号資産**: UTC ネイティブ
- **表示層**: ユーザー設定のタイムゾーンに変換（デフォルト Asia/Tokyo）

### 3.4 欠損値処理

| 状況 | 処理 |
|------|------|
| 休場日 | レコード自体を生成しない（forward-fill しない）。is_trading_day フラグで管理 |
| 一部ティッカーの取得失敗 | stale_data フラグを立て、前回値を保持。特徴量生成時に stale_count を追跡 |
| OHLCV 内の欠損 | 出来高 0 の場合は流動性フラグを立てる。価格 NaN は当該レコード除外 |
| マクロ指標の遅延公表 | point-in-time 原則: 公表日ベースで特徴量に反映。リビジョンは別カラムで管理 |

### 3.5 サバイバーシップバイアス対策

- S&P 500 構成銘柄リストは **時点ごとの構成** を使う。Wikipedia の過去履歴 or `sp500_historical_constituents` データセットを利用
- 上場廃止銘柄のデータも保持（delisted フラグ付与）
- バックテスト時は、当時のユニバースに基づいて銘柄選択する

### 3.6 将来情報混入 (Look-ahead Bias) 防止

| 対策 | 詳細 |
|------|------|
| point-in-time DB | 全データに `available_at` タイムスタンプを付与。特徴量生成は `available_at <= as_of_date` のデータのみ使用 |
| ラベル隔離 | ラベル (y) は特徴量 (X) と別テーブル/ファイルに保存。join は学習パイプラインの最後に実行 |
| walk-forward 強制 | TimeSeriesSplit のみ使用。通常の k-fold は禁止 |
| エンバーゴ期間 | 学習期間と評価期間の間に gap_days (デフォルト 2 日) を設ける。autocorrelation による情報漏洩を防ぐ [López de Prado 2018] |
| 特徴量監査 | 各特徴量に `max_lag` メタデータを付与。`max_lag < prediction_horizon` の特徴量は自動的にフラグ |

---

## 4. モデル設計

### 4.1 ベースラインモデル

| モデル | 用途 | 実装 |
|--------|------|------|
| Naive (前日と同方向) | 方向分類ベースライン | 自前 |
| Historical Mean Return | リターン回帰ベースライン | 自前 |
| GARCH(1,1) | ボラティリティベースライン | `arch` パッケージ |
| Buy & Hold | バックテスト比較用 | 自前 |

### 4.2 改良モデル

| モデル | 用途 | ライブラリ | 選定理由 |
|--------|------|-----------|---------|
| **LightGBM** (分類/回帰) | **本命**: 方向分類 + リターン回帰 | `lightgbm` | 表形式データでの実績、学習速度、特徴量重要度の解釈性。金融 ML の実務では GBM 系が Transformer 系を安定的に上回る報告が多い [Gu et al. 2020, "Empirical Asset Pricing via Machine Learning", RFS] |
| XGBoost | LightGBM との比較用 | `xgboost` | アンサンブル多様性確保 |
| Linear (Lasso/Ridge) | 特徴量選択の参考 | `scikit-learn` | 線形ベースラインとして。重要特徴量の確認 |
| LSTM (2層) | 時系列パターン捕捉 | `pytorch` | 非線形時系列依存の補完。ただし Phase 2 以降 |
| Transformer (PatchTST 系) | 実験枠 | `pytorch` | 長期依存捕捉の実験。本番投入は性能次第 |

**「最初の本命」: LightGBM**

理由:
1. 表形式特徴量（テクニカル + ファンダメンタル + マクロ）との親和性が高い
2. 学習・推論が高速（日次バッチで数百銘柄を数分で処理可能）
3. 特徴量重要度 (SHAP) が直接得られ、解釈性が高い
4. 過学習制御パラメータ（`num_leaves`, `min_data_in_leaf`, `lambda_l1/l2`）が豊富
5. 金融 ML の実務・学術の両面で実績がある [Gu et al. 2020]

### 4.3 特徴量一覧

| カテゴリ | 特徴量 | 数 | ラグ |
|---------|--------|---|------|
| リターン | log_return_{1,5,10,20}d | 4 | 0 |
| ボラティリティ | realized_vol_{5,10,20}d, garman_klass_vol | 4 | 0 |
| モメンタム | RSI_14, MACD, MACD_signal, ROC_{5,10,20} | 6 | 0 |
| 出来高 | volume_ratio_{5,20}d, OBV_slope_10d | 3 | 0 |
| 移動平均乖離 | price_vs_SMA_{5,20,50,200} | 4 | 0 |
| ボリンジャー | bb_position, bb_width | 2 | 0 |
| ATR | ATR_14, ATR_ratio | 2 | 0 |
| マクロ | fed_funds_rate, yield_curve_slope, VIX, VIX_change_5d | 4 | point-in-time |
| ファンダメンタル | PE_ratio, PB_ratio, earnings_surprise_last | 3 | 最新四半期 |
| センチメント | news_sentiment_score, news_volume_3d | 2 | 0 |
| カレンダー | day_of_week, month, is_month_end, days_to_earnings | 4 | 0 |
| セクター | sector_return_1d, sector_momentum_5d | 2 | 0 |
| **合計** | | **～40** | |

特徴量は段階的に追加する。Phase 1 ではテクニカル + リターン + 出来高のみ（約 20 特徴量）。

### 4.4 ラベル定義

```python
# 方向分類 (3 クラス)
def make_direction_label(returns: pd.Series, threshold: float = 0.005) -> pd.Series:
    return pd.cut(returns, bins=[-np.inf, -threshold, threshold, np.inf],
                  labels=[0, 1, 2])  # 0=DOWN, 1=FLAT, 2=UP

# リターン回帰
# target = log(close_t+5 / close_t)

# ボラティリティ
# target = realized_vol(close, window=20)  # t+1 ~ t+20 の実現ボラ
```

### 4.5 学習データ分割: Walk-Forward + Expanding Window

```
|---- 学習 (expanding) ----|- gap -|--- 評価 ---|
|==========================|  2d   |===========|
|==============================|  2d   |===========|
|==================================|  2d   |===========|
```

| パラメータ | 値 |
|-----------|-----|
| 初期学習期間 | 756 営業日（約 3 年） |
| 評価期間 | 63 営業日（約 3 ヶ月） |
| ステップ | 63 営業日 |
| エンバーゴ (gap) | 2 営業日 |
| 方式 | Expanding Window（データ蓄積の恩恵を受ける） |

**Combinatorial Purged Cross-Validation (CPCV)** [López de Prado 2018]:
ハイパーパラメータチューニング時に使用。通常の TimeSeriesSplit に加え、CPCV で Probability of Backtest Overfitting (PBO) を計算し、PBO > 0.5 のモデルは棄却する。

### 4.6 キャリブレーション

- 方向分類: Platt Scaling (sigmoid calibration) を walk-forward の各 fold で適用
- 校正検証: Reliability diagram + Expected Calibration Error (ECE)
- 校正が劣化したら再学習トリガー

### 4.7 モデル比較方法

- Walk-forward 全 fold での平均 + 標準偏差を報告
- Diebold-Mariano 検定でベースラインとの有意差を確認
- 最終選択は Information Ratio (仮想 L/S ポートフォリオ) で判断

### 4.8 レジーム検知

| 手法 | 実装 | 用途 |
|------|------|------|
| Hidden Markov Model (2-state) | `hmmlearn` | 高ボラ / 低ボラの判定 |
| VIX 閾値 | VIX > 25 → 高ストレス | 単純ルール併用 |
| 移動 Sharpe 比 | rolling_sharpe_60d | トレンド / レンジの判定 |

レジームに応じてモデルの重み配分を変える（例: 高ストレス時は予測信頼度を下げる、ボラティリティ予測の重みを上げる）。

---

## 5. 評価設計

### 5.1 バックテスト手順

```
1. データ取得 (as_of_date を指定し point-in-time で再現)
2. 特徴量生成 (feature_version タグ付き)
3. Walk-forward ループ:
   a. 学習データでモデル fit
   b. エンバーゴ期間スキップ
   c. 評価データで predict
   d. 予測を保存 (run_id, model_version, fold_id)
4. 全 fold の予測を結合
5. 指標計算: hit_rate, log_loss, IC, rank_IC, Brier
6. 仮想 L/S ポートフォリオ構築 → IR, max_dd, turnover
7. CPCV で PBO 計算
8. backtest_result JSON 出力
```

### 5.2 ペーパートレード手順

```
1. 日次スケジューラ (cron / Airflow) でパイプラインを起動
2. 最新データ取得 → 特徴量生成 → 推論
3. forecast_output JSON を生成
4. 仮想ポートフォリオに反映 (紙上のみ、発注なし)
5. 翌日に実績と照合 → accuracy_log に追記
6. ドリフト監視: PSI (Population Stability Index) を計算
7. 週次で performance_report 生成
```

### 5.3 取引コスト考慮

| コスト要素 | 株式 | 暗号資産 |
|-----------|------|---------|
| 手数料 | 片道 5 bps (保守的) | 片道 10 bps |
| スリッページ | 片道 5 bps | 片道 15 bps |
| マーケットインパクト | 大型株は無視。小型株は出来高の 1% 以内に制限 | 時価総額下位は除外 |
| 合計想定 | 往復 20 bps | 往復 50 bps |

バックテスト結果は**コスト控除後**で報告する。コスト控除前の数値のみで判断しない。

### 5.4 リスク指標

| 指標 | 計算方法 | 閾値 |
|------|---------|------|
| Maximum Drawdown | 累積リターンの最大谷 | < 15% |
| Sharpe Ratio (年率) | mean_ret / std_ret * sqrt(252) | > 0.5 (コスト控除後) |
| Sortino Ratio | mean_ret / downside_std * sqrt(252) | > 0.7 |
| Calmar Ratio | annual_ret / max_dd | > 0.5 |
| Turnover | 日次回転率 | < 30% / day |
| VaR (95%) | Historical VaR | 情報提供 |

### 5.5 ドリフト監視

| 監視対象 | 手法 | アクション |
|---------|------|-----------|
| 特徴量分布 | PSI (Population Stability Index) | PSI > 0.2 → WARNING, > 0.25 → 再学習検討 |
| 予測精度 | Rolling hit_rate (60 日窓) | hit_rate < baseline (0.33) が 20 日継続 → 再学習 |
| モデル出力分布 | 予測確率のエントロピー変化 | 急激な低エントロピー化 → 過信フラグ |
| データ品質 | 欠損率、stale_data 比率 | stale > 10% → データソース調査 |

### 5.6 再学習条件

以下のいずれかを満たしたら再学習をトリガーする:
1. 前回学習から 63 営業日（約 3 ヶ月）経過
2. PSI > 0.25 が 5 日連続
3. Rolling hit_rate < baseline が 20 日連続
4. レジーム変化を HMM が検出（状態遷移確率 > 0.7）

再学習は自動実行するが、**本番モデル入れ替え**は人間承認を必須とする（shadow deploy → 比較 → 承認 → swap）。

---

## 6. API / MCP / Tool 設計

### 6.1 外部 API 一覧と接続優先順位

| 優先度 | API | 役割 | 認証 | レート制限 | フォールバック |
|--------|-----|------|------|-----------|--------------|
| 1 | Polygon.io | 株価 OHLCV (主) | API Key | 5 req/min (free) | Alpha Vantage |
| 2 | Alpha Vantage (MCP) | 株価 OHLCV (副) + テクニカル | API Key | 25 req/day (free) | Yahoo Finance (非公式) |
| 3 | SEC EDGAR | 企業開示・XBRL | 不要 (User-Agent 必須) | 10 req/sec | — |
| 4 | FRED | マクロ指標 | API Key | 制限緩い | — |
| 5 | CoinGecko | 暗号資産価格 | 不要 (free) / API Key (paid) | 30 req/min | Binance REST |
| 6 | Finnhub | ニュース | API Key | 60 req/min | NewsAPI |
| 7 | Binance | 暗号資産 WS | API Key | 1200 weight/min | CoinGecko |

### 6.2 MCP 化すべきツール

| MCP Tool 名 | 対応 API | 方向 | 説明 |
|-------------|---------|------|------|
| `alpha_vantage_mcp` | Alpha Vantage | 読取専用 | **公式 MCP サーバー既存**。TIME_SERIES_DAILY, OVERVIEW, EARNINGS 等 |
| `market_data_fetcher` | Polygon.io | 読取専用 | OHLCV 取得 + キャッシュ。自前 MCP 化推奨 |
| `sec_filing_reader` | SEC EDGAR | 読取専用 | 10-K/10-Q セクション抽出 + XBRL パース |
| `macro_data_fetcher` | FRED | 読取専用 | マクロ系列取得 |
| `news_sentiment_tool` | Finnhub / LLM | 読取専用 | ニュース取得 + LLM センチメント分析 |
| `forecast_runner` | 内部 | 読取専用 | モデル推論実行 + forecast_output 生成 |
| `backtest_runner` | 内部 | 読取専用 | バックテスト実行 + backtest_result 生成 |

### 6.3 読み取り専用と書き込みの分離

```
READ-ONLY (全 MCP ツール):
  - データ取得
  - 特徴量生成
  - モデル推論
  - レポート生成

WRITE (制限付き、デフォルト無効):
  - order_executor: 発注機能。デフォルト disabled。
    有効化には config で enable_live_trading: true + 
    環境変数 CONFIRM_LIVE_TRADING=yes の両方が必要。
  - model_deployer: モデル入れ替え。human_approval: required。
```

### 6.4 JSON Schema で厳格化すべき I/O

全 MCP ツールの入出力を JSON Schema で定義する（セクション 7 参照）。特に:
- `forecast_runner` の出力は `forecast_output` schema に厳密に従う
- `backtest_runner` の出力は `backtest_result` schema に厳密に従う
- LLM 出力（センチメント等）は structured output + validation で強制

---

## 7. 出力スキーマ

### 7.1 forecast_output

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "ForecastOutput",
  "type": "object",
  "required": ["forecast_id", "generated_at", "model_version", "horizon", "predictions"],
  "properties": {
    "forecast_id": {
      "type": "string",
      "format": "uuid",
      "description": "一意の予測識別子"
    },
    "generated_at": {
      "type": "string",
      "format": "date-time",
      "description": "予測生成時刻 (UTC)"
    },
    "model_version": {
      "type": "string",
      "pattern": "^v[0-9]+\\.[0-9]+\\.[0-9]+$",
      "description": "モデルバージョン (semver)"
    },
    "horizon": {
      "type": "string",
      "enum": ["1d", "5d", "20d", "event"],
      "description": "予測ホライズン"
    },
    "regime": {
      "type": "string",
      "enum": ["low_vol", "high_vol", "transition", "unknown"],
      "description": "検出されたマーケットレジーム"
    },
    "predictions": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["ticker", "direction", "probabilities", "confidence", "stale_data_flag"],
        "properties": {
          "ticker": { "type": "string" },
          "direction": {
            "type": "string",
            "enum": ["UP", "DOWN", "FLAT"]
          },
          "probabilities": {
            "type": "object",
            "required": ["UP", "DOWN", "FLAT"],
            "properties": {
              "UP": { "type": "number", "minimum": 0, "maximum": 1 },
              "DOWN": { "type": "number", "minimum": 0, "maximum": 1 },
              "FLAT": { "type": "number", "minimum": 0, "maximum": 1 }
            }
          },
          "expected_return": {
            "type": ["number", "null"],
            "description": "期待リターン (5d horizon の場合)"
          },
          "predicted_volatility": {
            "type": ["number", "null"],
            "description": "予測ボラティリティ (20d horizon の場合)"
          },
          "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
            "description": "最大確率 > 0.5: high, > 0.4: medium, else: low"
          },
          "top_features": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "name": { "type": "string" },
                "shap_value": { "type": "number" }
              }
            },
            "maxItems": 5,
            "description": "SHAP 上位 5 特徴量"
          },
          "stale_data_flag": {
            "type": "boolean",
            "description": "入力データに stale がある場合 true"
          }
        }
      }
    }
  }
}
```

### 7.2 evidence_bundle

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "EvidenceBundle",
  "type": "object",
  "required": ["bundle_id", "forecast_id", "created_at", "data_snapshot", "model_info"],
  "properties": {
    "bundle_id": { "type": "string", "format": "uuid" },
    "forecast_id": { "type": "string", "format": "uuid" },
    "created_at": { "type": "string", "format": "date-time" },
    "data_snapshot": {
      "type": "object",
      "required": ["ohlcv_as_of", "features_hash", "stale_tickers"],
      "properties": {
        "ohlcv_as_of": { "type": "string", "format": "date-time" },
        "macro_as_of": { "type": "string", "format": "date-time" },
        "news_as_of": { "type": "string", "format": "date-time" },
        "features_hash": {
          "type": "string",
          "description": "特徴量マトリクスの SHA-256"
        },
        "stale_tickers": {
          "type": "array",
          "items": { "type": "string" }
        }
      }
    },
    "model_info": {
      "type": "object",
      "required": ["model_name", "model_version", "trained_at", "training_samples"],
      "properties": {
        "model_name": { "type": "string" },
        "model_version": { "type": "string" },
        "trained_at": { "type": "string", "format": "date-time" },
        "training_samples": { "type": "integer" },
        "hyperparameters": { "type": "object" }
      }
    },
    "news_summary": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "headline": { "type": "string" },
          "source": { "type": "string" },
          "published_at": { "type": "string", "format": "date-time" },
          "sentiment_score": { "type": "number", "minimum": -1, "maximum": 1 },
          "relevance_score": { "type": "number", "minimum": 0, "maximum": 1 }
        }
      },
      "maxItems": 20
    }
  }
}
```

### 7.3 risk_review

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "RiskReview",
  "type": "object",
  "required": ["review_id", "forecast_id", "reviewed_at", "checks", "overall_risk_level", "approval"],
  "properties": {
    "review_id": { "type": "string", "format": "uuid" },
    "forecast_id": { "type": "string", "format": "uuid" },
    "reviewed_at": { "type": "string", "format": "date-time" },
    "checks": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["check_name", "status", "detail"],
        "properties": {
          "check_name": {
            "type": "string",
            "enum": [
              "data_freshness",
              "feature_drift",
              "prediction_concentration",
              "model_staleness",
              "liquidity_check",
              "regime_consistency",
              "hallucination_guard",
              "source_integrity"
            ]
          },
          "status": {
            "type": "string",
            "enum": ["PASS", "WARNING", "FAIL"]
          },
          "detail": { "type": "string" },
          "metric_value": { "type": ["number", "null"] },
          "threshold": { "type": ["number", "null"] }
        }
      }
    },
    "overall_risk_level": {
      "type": "string",
      "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    },
    "approval": {
      "type": "string",
      "enum": ["AUTO_APPROVED", "MANUAL_REVIEW_REQUIRED", "BLOCKED"],
      "description": "FAIL が 1 つでもあれば BLOCKED"
    },
    "recommendations": {
      "type": "array",
      "items": { "type": "string" }
    }
  }
}
```

### 7.4 report_payload

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "ReportPayload",
  "type": "object",
  "required": ["report_id", "report_type", "generated_at", "summary", "sections"],
  "properties": {
    "report_id": { "type": "string", "format": "uuid" },
    "report_type": {
      "type": "string",
      "enum": ["daily_forecast", "weekly_review", "backtest_report", "drift_alert"]
    },
    "generated_at": { "type": "string", "format": "date-time" },
    "summary": {
      "type": "string",
      "maxLength": 500,
      "description": "要約 (LLM 生成、ファクトチェック済み)"
    },
    "sections": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["title", "content"],
        "properties": {
          "title": { "type": "string" },
          "content": { "type": "string" },
          "charts": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "chart_type": {
                  "type": "string",
                  "enum": ["line", "bar", "heatmap", "scatter", "table"]
                },
                "data_ref": {
                  "type": "string",
                  "description": "チャート用データファイルへのパス"
                },
                "title": { "type": "string" }
              }
            }
          }
        }
      }
    },
    "forecast_ids": {
      "type": "array",
      "items": { "type": "string", "format": "uuid" }
    },
    "risk_review_id": { "type": "string", "format": "uuid" },
    "disclaimers": {
      "type": "array",
      "items": { "type": "string" },
      "description": "免責事項。必ず含める"
    }
  }
}
```

### 7.5 backtest_result

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "BacktestResult",
  "type": "object",
  "required": ["backtest_id", "config", "completed_at", "folds", "aggregate_metrics", "cost_adjusted_metrics"],
  "properties": {
    "backtest_id": { "type": "string", "format": "uuid" },
    "config": {
      "type": "object",
      "required": ["model_name", "model_version", "start_date", "end_date", "horizon", "initial_train_days", "eval_days", "embargo_days"],
      "properties": {
        "model_name": { "type": "string" },
        "model_version": { "type": "string" },
        "start_date": { "type": "string", "format": "date" },
        "end_date": { "type": "string", "format": "date" },
        "horizon": { "type": "string" },
        "initial_train_days": { "type": "integer" },
        "eval_days": { "type": "integer" },
        "embargo_days": { "type": "integer" },
        "feature_version": { "type": "string" },
        "universe": { "type": "string" }
      }
    },
    "completed_at": { "type": "string", "format": "date-time" },
    "folds": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["fold_id", "train_start", "train_end", "eval_start", "eval_end", "metrics"],
        "properties": {
          "fold_id": { "type": "integer" },
          "train_start": { "type": "string", "format": "date" },
          "train_end": { "type": "string", "format": "date" },
          "eval_start": { "type": "string", "format": "date" },
          "eval_end": { "type": "string", "format": "date" },
          "n_train": { "type": "integer" },
          "n_eval": { "type": "integer" },
          "metrics": {
            "type": "object",
            "properties": {
              "hit_rate": { "type": "number" },
              "log_loss": { "type": "number" },
              "brier_score": { "type": "number" },
              "ic": { "type": "number" },
              "rank_ic": { "type": "number" }
            }
          }
        }
      }
    },
    "aggregate_metrics": {
      "type": "object",
      "description": "全 fold の平均 ± 標準偏差",
      "properties": {
        "hit_rate_mean": { "type": "number" },
        "hit_rate_std": { "type": "number" },
        "log_loss_mean": { "type": "number" },
        "ic_mean": { "type": "number" },
        "rank_ic_mean": { "type": "number" }
      }
    },
    "cost_adjusted_metrics": {
      "type": "object",
      "description": "取引コスト控除後の仮想 L/S 指標",
      "properties": {
        "information_ratio": { "type": "number" },
        "sharpe_ratio": { "type": "number" },
        "sortino_ratio": { "type": "number" },
        "max_drawdown": { "type": "number" },
        "calmar_ratio": { "type": "number" },
        "annual_return": { "type": "number" },
        "annual_volatility": { "type": "number" },
        "avg_daily_turnover": { "type": "number" },
        "total_cost_bps": { "type": "number" }
      }
    },
    "pbo": {
      "type": ["number", "null"],
      "description": "Probability of Backtest Overfitting (CPCV)"
    },
    "diebold_mariano": {
      "type": "object",
      "properties": {
        "vs_baseline": { "type": "string" },
        "dm_statistic": { "type": "number" },
        "p_value": { "type": "number" }
      }
    }
  }
}
```

---

## 8. リスクとガードレール

### 8.1 データリーク

| リスク | 対策 | 検出方法 |
|--------|------|---------|
| 将来の価格がラベルだけでなく特徴量にも混入 | point-in-time DB + `available_at` 制約 | 特徴量の `max_lag` 自動監査スクリプト |
| 学習/評価データの時間的重複 | エンバーゴ期間 (2日) | walk-forward split の overlap チェック |
| ターゲットエンコーディング等での漏洩 | ターゲットエンコーディング禁止 (初期)。使う場合は fold 内のみ | コードレビュー + 自動テスト |

### 8.2 過学習

| リスク | 対策 | 検出方法 |
|--------|------|---------|
| 特徴量数 >> サンプル数 | 初期特徴量を ～40 に制限。L1 正則化で絞る | train-eval gap の監視 |
| ハイパーパラメータのチェリーピッキング | CPCV で PBO 計算。PBO > 0.5 → 棄却 | PBO レポート |
| 過度なバックテスト反復 | バックテスト回数を記録。Deflated Sharpe Ratio で調整 | DSR < 2.0 → 棄却 |

### 8.3 Regime Shift

| リスク | 対策 | 検出方法 |
|--------|------|---------|
| 過去にないマーケット環境 | HMM レジーム検知 + 予測信頼度の動的調整 | PSI > 0.25, HMM 状態遷移 |
| 金融危機・ブラックスワン | 高ストレス時はポジションサイズ自動縮小。予測を保守的に | VIX > 35 → 信頼度強制 "low" |
| 構造変化 (規制変更、市場制度変更) | 再学習トリガー + 手動レビュー | ニュースフィルタで制度変更検知 |

### 8.4 ハルシネーション

| リスク | 対策 |
|--------|------|
| LLM がセンチメント分析で存在しない情報を生成 | ニュース原文を必ず添付。LLM 出力には source_url 必須。source_url が検証不能なら出力除外 |
| レポート生成時の数値捏造 | Report Agent は forecast_output + evidence_bundle の数値のみ引用可能。自由記述部分は「所見」と明示 |
| 企業名・ティッカーの取り違え | LLM 入力にティッカー + 正式社名を必ず含める。出力のティッカーを入力と照合 |

### 8.5 Prompt Injection

| リスク | 対策 |
|--------|------|
| ニュース本文に悪意ある指示が含まれる | ニュース入力を LLM に渡す際、system prompt で「ニュース分析のみ行え」と制約。ユーザー入力とデータ入力を分離 |
| API レスポンスへの injection | API レスポンスを LLM に直接渡さない。構造化データ (JSON) として処理 |

### 8.6 外部ソース汚染

| リスク | 対策 |
|--------|------|
| ニュースソースのフェイクニュース | 複数ソース照合。単一ソースのみの場合は信頼度を下げる |
| API データ改ざん | HTTPS 通信 + レスポンスの妥当性チェック (前日比 ±50% 超は異常フラグ) |

### 8.7 API 障害

| リスク | 対策 |
|--------|------|
| 主データソース停止 | フォールバック構成 (Polygon → Alpha Vantage → Yahoo Finance)。自動切替 |
| レート制限超過 | Exponential backoff + キューイング。リクエスト数を budget として管理 |
| 全データソース停止 | stale_data フラグ付きで前回データを使用。予測信頼度を "low" に強制。24h 以上 stale なら予測停止 |

### 8.8 Stale Data

| リスク | 対策 |
|--------|------|
| 古いデータで予測実行 | 各データに `fetched_at` タイムスタンプ。`fetched_at` と現在時刻の差が閾値超 → stale フラグ |
| 閾値 | OHLCV: 2h, マクロ: 48h, ニュース: 6h |

### 8.9 誤発注防止

| 対策 | 詳細 |
|------|------|
| デフォルト無効 | `enable_live_trading: false` (config) + `CONFIRM_LIVE_TRADING` 環境変数 |
| ダブルチェック | 発注前に risk_review の approval が AUTO_APPROVED であることを検証 |
| ポジション制限 | 1 銘柄あたり最大 5% (総資産比)。1 日の最大発注回数 50 |
| キルスイッチ | `EMERGENCY_STOP` 環境変数 or Redis key で即時全停止 |
| ドライラン | 発注モジュールは必ず dry_run モードを持つ。本番切替は設定ファイル変更 + 人間承認 |

### 8.10 法務・コンプライアンス

| 観点 | 注意事項 |
|------|---------|
| 投資助言規制 | 本システムの出力は「分析ツールの結果」であり投資助言ではない。免責事項を全レポートに付与 |
| SEC EDGAR 利用規約 | User-Agent に連絡先を含める (SEC 要件)。10 req/sec 制限を厳守 |
| データ再配布 | API データの再配布は各 API の利用規約に従う。内部利用のみ |
| 個人情報 | 本システムは個人情報を扱わない |
| 暗号資産規制 | 管轄によって異なる。発注機能を有効化する場合は現地規制を確認 |

---

## 9. 実装ハンドオフ仕様

### 9.1 推奨ディレクトリ構成

```
market-prediction-agent/
├── pyproject.toml              # プロジェクト定義 (uv / poetry)
├── config/
│   ├── default.yaml            # デフォルト設定
│   ├── dev.yaml                # 開発環境オーバーライド
│   └── schemas/                # JSON Schema ファイル
│       ├── forecast_output.json
│       ├── evidence_bundle.json
│       ├── risk_review.json
│       ├── report_payload.json
│       └── backtest_result.json
├── src/
│   ├── agents/
│   │   ├── research/           # Research Agent
│   │   │   ├── news_fetcher.py
│   │   │   ├── filing_reader.py
│   │   │   └── sentiment.py
│   │   ├── data/               # Data Agent
│   │   │   ├── ohlcv.py
│   │   │   ├── macro.py
│   │   │   ├── crypto.py
│   │   │   └── normalizer.py
│   │   ├── forecast/           # Forecast Agent
│   │   │   ├── features.py
│   │   │   ├── labels.py
│   │   │   ├── models/
│   │   │   │   ├── lightgbm_model.py
│   │   │   │   ├── xgboost_model.py
│   │   │   │   ├── linear_model.py
│   │   │   │   └── base.py     # 共通インターフェース
│   │   │   ├── walk_forward.py
│   │   │   ├── calibration.py
│   │   │   └── regime.py
│   │   ├── risk/               # Risk/Critic Agent
│   │   │   ├── checks.py
│   │   │   ├── drift.py
│   │   │   └── guardrails.py
│   │   └── report/             # Report Agent
│   │       ├── generator.py
│   │       └── templates/
│   ├── mcp/                    # MCP ツール定義
│   │   ├── market_data.py
│   │   ├── sec_reader.py
│   │   └── macro_data.py
│   ├── storage/                # データ永続化
│   │   ├── parquet_store.py
│   │   └── model_registry.py
│   ├── pipeline/               # オーケストレーション
│   │   ├── daily_batch.py
│   │   ├── backtest.py
│   │   └── scheduler.py
│   └── common/                 # 共通ユーティリティ
│       ├── config.py
│       ├── logging.py
│       ├── schemas.py          # JSON Schema バリデーション
│       └── time_utils.py
├── tests/
│   ├── unit/
│   │   ├── test_features.py
│   │   ├── test_labels.py
│   │   ├── test_walk_forward.py
│   │   ├── test_checks.py
│   │   └── test_schemas.py
│   ├── integration/
│   │   ├── test_data_pipeline.py
│   │   ├── test_forecast_pipeline.py
│   │   └── test_api_connectivity.py
│   └── fixtures/
│       ├── dummy_ohlcv.parquet
│       ├── dummy_features.parquet
│       └── dummy_forecast.json
├── scripts/
│   ├── generate_dummy_data.py
│   ├── run_backtest.py
│   └── validate_schemas.py
└── storage/                    # データ保存先 (.gitignore 対象)
    ├── raw/
    ├── processed/
    ├── models/
    └── outputs/
```

### 9.2 推奨言語・ライブラリ

| カテゴリ | ライブラリ | バージョン | 用途 |
|---------|-----------|-----------|------|
| 言語 | Python | ≥ 3.11 | — |
| パッケージ管理 | uv | latest | 高速依存解決 |
| データ操作 | pandas, polars | ≥ 2.0, ≥ 0.20 | DataFrame 操作。polars は大量データ用 |
| ストレージ | pyarrow | ≥ 14.0 | Parquet I/O |
| ML: GBM | lightgbm | ≥ 4.0 | 本命モデル |
| ML: GBM 比較 | xgboost | ≥ 2.0 | 比較用 |
| ML: 線形 | scikit-learn | ≥ 1.4 | Lasso/Ridge + 評価指標 |
| ML: 深層学習 | pytorch | ≥ 2.0 | LSTM / Transformer (Phase 2+) |
| テクニカル | pandas-ta | ≥ 0.3 | テクニカル指標計算 |
| ボラティリティ | arch | ≥ 6.0 | GARCH |
| レジーム | hmmlearn | ≥ 0.3 | HMM |
| 解釈性 | shap | ≥ 0.44 | SHAP 値 |
| 校正 | — | scikit-learn | CalibratedClassifierCV |
| スキーマ検証 | jsonschema | ≥ 4.20 | JSON Schema validation |
| 設定 | pydantic | ≥ 2.0 | 型安全な設定管理 |
| HTTP | httpx | ≥ 0.27 | 非同期 HTTP クライアント |
| キュー | redis (+ redis-py) | ≥ 5.0 | エージェント間通信 (オプション) |
| LLM | anthropic | latest | Claude API (Research/Report Agent) |
| テスト | pytest, pytest-cov | ≥ 8.0 | テスト + カバレッジ |
| 型チェック | mypy | ≥ 1.8 | 静的型検査 |
| Lint | ruff | latest | Linting + Formatting |

### 9.3 モジュール分割方針

各エージェントは独立したサブパッケージとする。依存関係:

```
common ← storage ← agents/data ← agents/forecast
                  ← agents/research
                  ← agents/risk (depends on forecast output)
                  ← agents/report (depends on all outputs)
pipeline ← all agents
mcp ← agents/data, agents/forecast
```

循環依存禁止。エージェント間の通信は JSON (output schema) 経由のみ。

### 9.4 設定ファイル設計

```yaml
# config/default.yaml
app:
  name: market-prediction-agent
  version: "0.1.0"
  log_level: INFO

data:
  primary_source: polygon
  fallback_source: alphavantage
  universe: sp500
  ohlcv_granularity: daily
  storage_path: ./storage
  stale_threshold_hours:
    ohlcv: 2
    macro: 48
    news: 6

model:
  primary: lightgbm
  direction_threshold: 0.005
  walk_forward:
    initial_train_days: 756
    eval_days: 63
    step_days: 63
    embargo_days: 2
  calibration: platt
  retrain_interval_days: 63

risk:
  psi_warning: 0.20
  psi_critical: 0.25
  max_drawdown_limit: 0.15
  vix_stress_threshold: 25
  vix_extreme_threshold: 35

trading:
  enable_live_trading: false
  max_position_pct: 0.05
  max_daily_orders: 50
  cost_bps:
    equity_oneway: 5
    crypto_oneway: 10
  slippage_bps:
    equity_oneway: 5
    crypto_oneway: 15

llm:
  provider: anthropic
  model: claude-sonnet-4-6
  max_monthly_cost_usd: 50
  temperature: 0.1

api_keys:
  # 環境変数から読み込む (設定ファイルには書かない)
  polygon: ${POLYGON_API_KEY}
  alphavantage: ${ALPHAVANTAGE_API_KEY}
  fred: ${FRED_API_KEY}
  finnhub: ${FINNHUB_API_KEY}
  coingecko: ${COINGECKO_API_KEY}
  binance_key: ${BINANCE_API_KEY}
  binance_secret: ${BINANCE_API_SECRET}
  anthropic: ${ANTHROPIC_API_KEY}
```

### 9.5 環境変数一覧

| 変数名 | 必須 | 説明 |
|--------|------|------|
| `POLYGON_API_KEY` | Yes (Phase 0) | Polygon.io API キー |
| `ALPHAVANTAGE_API_KEY` | Yes | Alpha Vantage API キー (MCP 用) |
| `FRED_API_KEY` | Yes | FRED API キー |
| `FINNHUB_API_KEY` | Yes (Phase 2) | Finnhub ニュース API キー |
| `COINGECKO_API_KEY` | No (free tier は不要) | CoinGecko Pro API キー |
| `BINANCE_API_KEY` | No (Phase 4) | Binance API キー |
| `BINANCE_API_SECRET` | No (Phase 4) | Binance API シークレット |
| `ANTHROPIC_API_KEY` | Yes (Phase 2) | Claude API キー |
| `CONFIRM_LIVE_TRADING` | No | "yes" で発注有効化 (デフォルト無効) |
| `EMERGENCY_STOP` | No | "true" で即時全停止 |
| `CONFIG_PATH` | No | 設定ファイルパス (デフォルト `config/default.yaml`) |
| `LOG_LEVEL` | No | ログレベル (デフォルト INFO) |

### 9.6 最小実装順序

```
Phase 0 — データ基盤 (Week 1-2)
  0.1  pyproject.toml + ディレクトリ構成 + CI 設定
  0.2  config 読み込み (pydantic)
  0.3  OHLCV 取得 (Polygon.io REST → Parquet 保存)
  0.4  OHLCV 取得フォールバック (Alpha Vantage)
  0.5  マクロデータ取得 (FRED)
  0.6  ダミーデータ生成スクリプト
  0.7  ユニットテスト: データ取得 + スキーマ検証

Phase 1 — 予測パイプライン (Week 3-4)
  1.1  テクニカル特徴量生成 (pandas-ta)
  1.2  ラベル生成 (direction 3-class)
  1.3  Walk-forward splitter 実装
  1.4  LightGBM モデル (学習 + 推論)
  1.5  forecast_output JSON 生成 + スキーマ検証
  1.6  バックテスト実行スクリプト
  1.7  backtest_result JSON 生成
  1.8  ベースラインモデル (naive) + 比較

Phase 2 — センチメント + ファンダメンタル (Week 5-6)
  2.1  Finnhub ニュース取得
  2.2  LLM センチメント分析 (Claude API)
  2.3  SEC EDGAR ファイリング取得
  2.4  ファンダメンタル特徴量追加
  2.5  マクロ特徴量追加
  2.6  全特徴量でモデル再学習 + 評価

Phase 3 — リスク + レポート (Week 7-8)
  3.1  Risk/Critic Agent: チェック項目実装
  3.2  ドリフト監視 (PSI)
  3.3  レジーム検知 (HMM)
  3.4  risk_review JSON 生成
  3.5  Report Agent: レポート生成 (LLM)
  3.6  evidence_bundle 生成
  3.7  統合テスト

Phase 4 — 運用化 (Week 9-10)
  4.1  日次バッチスケジューラ
  4.2  ペーパートレード機能
  4.3  暗号資産データ統合 (CoinGecko)
  4.4  MCP ツール化
  4.5  ドリフト監視自動化
  4.6  ドキュメント整備
```

### 9.7 テスト戦略

| レイヤー | 対象 | ツール | カバレッジ目標 |
|---------|------|--------|-------------|
| Unit | 特徴量生成、ラベル生成、スキーマ検証、時間関数 | pytest | 80% |
| Integration | データ取得 → 特徴量 → モデル → 出力の一気通貫 | pytest + fixtures | 主要パス |
| Schema | 全 JSON 出力が schema に適合するか | jsonschema + pytest | 100% |
| Regression | バックテスト結果の再現性 (seed 固定) | pytest + snapshot | 主要モデル |
| API Mock | 外部 API のモック (respx / responses) | pytest + respx | 全 API |

### 9.8 ダミーデータでの検証方法

`scripts/generate_dummy_data.py` で以下を生成:
- 100 銘柄 × 1000 営業日分の OHLCV (ランダムウォーク + ドリフト)
- 対応する特徴量マトリクス
- マクロ指標 10 系列
- ダミーニュースヘッドライン 500 件

検証手順:
1. ダミーデータでパイプライン全体を end-to-end 実行
2. 出力 JSON が全 schema に適合することを確認
3. ダミーデータ上のモデル性能がランダム付近 (hit_rate ≈ 0.33) であることを確認 → ランダムデータで高精度なら実装バグの可能性

### 9.9 未確定事項一覧

| # | 項目 | 現状 | 決定時期 |
|---|------|------|---------|
| U1 | Polygon.io 有料プランへの移行要否 | 無料枠 (5 req/min) で開始。不足なら有料化 | Phase 0 完了後 |
| U2 | 暗号資産の具体的なユニバース | BTC, ETH + 時価総額上位 20 (変動あり) | Phase 4 開始時 |
| U3 | LLM モデル選択 (Sonnet vs Haiku) | Sonnet 4.6 で開始。コスト次第で Haiku に切替 | Phase 2 |
| U4 | Redis vs ファイルベースのエージェント通信 | Phase 0-1 はファイルベース。Phase 4 で Redis 検討 | Phase 3 完了後 |
| U5 | クラウドデプロイ先 | ローカル開発優先。GCP / AWS は将来検討 | Phase 4 以降 |
| U6 | 日本株対応の優先度 | P2。米国株 + 暗号資産が安定してから | Phase 4 以降 |
| U7 | 分足データの使用頻度 | 日次モデルが主。分足は Phase 4 のリアルタイム用 | Phase 4 |

---

## 10. Codex 向け実装ブリーフ

```markdown
# 実装ブリーフ: Market Prediction Agent — Phase 0

## タスク
市場解析・予測エージェント基盤の Phase 0 (データ基盤) を実装する。

## 作業ディレクトリ
C:/Users/mizuki/market-prediction-agent/ (新規作成)

## 生成物
1. pyproject.toml — Python ≥ 3.11, uv 管理。依存: pandas, polars, pyarrow, httpx, 
   pydantic, lightgbm, jsonschema, pytest, ruff, mypy
2. config/default.yaml — 上記仕様書 §9.4 の内容
3. config/schemas/*.json — 上記仕様書 §7 の 5 つの JSON Schema ファイル
4. src/common/config.py — Pydantic で default.yaml を読み込む。環境変数展開あり
5. src/common/time_utils.py — UTC 変換、営業日判定、point-in-time フィルタ
6. src/common/schemas.py — jsonschema で出力を検証する validate() 関数
7. src/agents/data/ohlcv.py — Polygon.io REST で日次 OHLCV を取得 → Parquet 保存
   - フォールバック: Alpha Vantage
   - stale_data フラグ付与
   - 主キー: (ticker, timestamp_utc)
8. src/agents/data/macro.py — FRED API で主要マクロ系列を取得 → Parquet 保存
   - 系列: FEDFUNDS, T10Y2Y, VIXCLS, CPIAUCSL, UNRATE
   - point-in-time: available_at カラム付与
9. src/agents/data/normalizer.py — raw データを processed に変換。型統一、UTC 変換
10. scripts/generate_dummy_data.py — 100 銘柄 × 1000 日のダミー OHLCV + マクロ
11. tests/unit/test_schemas.py — 5 つの出力スキーマの検証テスト
12. tests/unit/test_time_utils.py — 時間ユーティリティのテスト
13. tests/integration/test_data_pipeline.py — ダミーデータでの E2E テスト
14. .gitignore — storage/, .env, __pycache__, *.pyc

## 完了条件
- [ ] `uv sync` で依存が解決される
- [ ] `pytest tests/` が全パス
- [ ] ダミーデータ生成 → OHLCV 取得 (mock) → Parquet 保存 → 読み込みの E2E が動作
- [ ] 5 つの JSON Schema がそれぞれ valid なサンプルで検証パス
- [ ] `ruff check src/` でエラーなし
- [ ] `mypy src/` でエラーなし
- [ ] config 内の API キーは環境変数経由のみ (ハードコード禁止)

## 禁止事項
- API キー、シークレットのハードコード
- 外部 API への実際のリクエスト (テストでは respx でモック)
- enable_live_trading のデフォルト true
- 仕様書 §3.6 の look-ahead bias 防止策に違反する実装

## テスト実行コマンド
uv run pytest tests/ -v --cov=src --cov-report=term-missing

## 参考
仕様書: C:/Users/mizuki/kb/market-prediction-agent-spec.md
```

---

## Open Questions

1. **データ量とストレージ**: S&P 500 全銘柄の日次データ 20 年分は ～50MB (Parquet)。分足データは ～50GB。分足の保存期間をどこまでにするか要判断
2. **LLM コスト最適化**: センチメント分析を Claude Sonnet でやるか Haiku でやるか。Haiku はコスト 1/10 だが精度への影響は要検証
3. **マルチアセット相関**: 株式と暗号資産の相関が変化する局面でのモデルの振る舞い。別モデル vs 統合モデルの検討
4. **リバランス頻度**: 日次リバランスは取引コストが嵩む。週次の方が現実的かもしれないが、予測ホライズンとの整合性が要検討
5. **CPCV 実装の計算コスト**: 銘柄数 × fold 数 × パラメータ候補数で計算量が膨大になる可能性。サンプリングや並列化が必要か

## Assumptions

1. **無料枠で開始**: 全 API は無料枠で開始し、ボトルネックが判明してから有料化する
2. **ローカル実行**: Phase 0-3 はローカルマシン (Windows 11) で開発・実行。GPU は不要 (LightGBM は CPU で十分)
3. **Python 単一言語**: フロントエンド不要。CLI + JSON 出力。将来の UI は別プロジェクトとして切り出す
4. **投資助言ではない**: 全レポートに免責事項を付与。本システムは分析ツールであり投資助言を行わない
5. **日本時間基準の運用**: スケジューラは Asia/Tokyo 基準で設定（米国市場は日本時間早朝に閉まる）
6. **再現性最優先**: ランダムシードの固定、データバージョニング、モデルのシリアライゼーションを徹底
7. **段階的な特徴量追加**: Phase 1 はテクニカル特徴量のみ。全特徴量の一括投入はしない
8. **エージェント通信は Phase 0-1 ではファイルベース**: JSON ファイルを介して連携。Redis は Phase 4 で検討

---

## 主要出典

- [López de Prado 2018] Marcos López de Prado, *Advances in Financial Machine Learning*, Wiley. — Walk-forward validation, CPCV, エンバーゴ、PBO の基本設計
- [Gu et al. 2020] Shihao Gu, Bryan Kelly, Dacheng Xiu, "Empirical Asset Pricing via Machine Learning", *Review of Financial Studies* 33(5), 2020. DOI: 10.1093/rfs/hhaa009 — GBM 系モデルの金融応用での優位性
- [Bailey & López de Prado 2014] David H. Bailey, Marcos López de Prado, "The Probability of Backtest Overfitting", *Journal of Computational Finance*. — PBO, Deflated Sharpe Ratio
- [Alpha Vantage MCP] https://mcp.alphavantage.co/ — 公式 MCP サーバー仕様
- [SEC EDGAR API] https://www.sec.gov/search-filings/edgar-application-programming-interfaces — 公式 API ドキュメント
- [FRED API] https://fred.stlouisfed.org/docs/api/fred/ — 公式 API ドキュメント
- [Polygon.io] https://polygon.io/ — 株価データ API
- [CoinGecko API] https://www.coingecko.com/en/api — 暗号資産データ API
