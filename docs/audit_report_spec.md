# Audit Report Spec

`render_audit_report.py` は既存 artifact を再計算せずに読み込み、review-friendly な `audit_report` JSON と Markdown を生成する。

## 対象 artifact

- `monitor_audit`
- `monitor_audit_suite`
- `weekly_review`
- `backtest_result`
- `paper_trading_batch`

## 標準見出し

Markdown は常に以下の順で出す。

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

## 表示ルール

- raw signal と effective trigger を分けて表示する
- candidate-level PBO と cluster-adjusted PBO を並記する
- `full_light` は routine monitoring、`full` は replay / deep-dive として明記する
- transition は `stable_transition` / `watch_transition` / `unstable_transition` を区別する
- 原因は `PBO` / `calibration` / `drift` / `regime` の 4 系統で整理する
- watch-only は effective action と混同しない

## 出力先

既定では以下へ保存する。

- `storage/outputs/audit_reports/<report_type>/<YYYY-MM-DD>/<report_id>.json`
- `storage/outputs/audit_reports/<report_type>/<YYYY-MM-DD>/<report_id>.md`
