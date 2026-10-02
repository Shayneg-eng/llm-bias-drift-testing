# LLM Bias & Drift Testing

A test harness for measuring **bias, asymmetry, and drift** in large language models under
repeated and adversarial prompting — does a model treat equivalent prompts differently, and do
its responses drift over a long interaction?

## Layout
| File | Role |
|---|---|
| `agent_loop.py` | Drives multi-turn interactions with a model |
| `batch_runner.py` | Runs batches of tests across conditions |
| `bias_analyzer_plot.py` / `plot_results.py` | Scoring + visualization |
| `past_tests/`, `past_reports/`, `batch_reports/` | Saved runs and write-ups |

Example outputs: `shayne_asymmetry.png`, `shayne_drift_curves.png`, `shayne_heatmap.png`.

## Setup
```bash
python -m pip install matplotlib numpy openai
export DEEPSEEK_API_KEY="sk-..."   # see .env.example
python batch_runner.py
```
