# Energy Consumption Forecaster

Forecasting daily electricity consumption for Germany using machine learning.
Built to practice forecasting methods I used at enercity, where I worked on consumption
models across a portfolio of several thousand customers.

## What it does

- **Predict** daily electricity consumption via a FastAPI endpoint
- **Auto-fetch** live grid data from SMARD (Bundesnetzagentur) and weather from Open-Meteo
- **Chat** with the forecaster in natural language using a LangChain agent
- **Explain** predictions with SHAP and LIME (model-agnostic XAI)
- **Flag** suspicious predictions with automated plausibility checks

## Data

Training combines two sources automatically — no manual download needed:

- **OPSD** — daily consumption 2006–2017 ([open-power-system-data.org](https://open-power-system-data.org))
- **SMARD** — live daily grid load 2018–present, streamed via API ([smard.de](https://www.smard.de))

Running `python3 src/train.py` fetches the latest SMARD data, merges it with OPSD,
and trains on ~7,300+ rows.

## Models

Trained on 2006–2024, tested on 2025:

| Model | MAE | RMSE |
| --- | --- | --- |
| Day-of-week baseline | 48 GWh | 78 GWh |
| KNN | 24 GWh | 33 GWh |
| MLP | 21 GWh | 31 GWh |

KNN is the production model — fast to train, near instant inference, and close to MLP accuracy.

### Christmas Day validation (vs actual SMARD data)

| Year | Predicted | Actual | Error | Temp |
| --- | --- | --- | --- | --- |
| 2023 | 1,073 GWh | 1,077 GWh | -0.4% | 9.3°C |
| 2024 | 1,071 GWh | 1,073 GWh | -0.2% | 2.5°C |
| 2025 | 1,072 GWh | 1,160 GWh | -7.6% | -7.1°C |

The 2025 error was caused by the model ignoring temperature on holidays — it learned
"holiday = low" regardless of weather. Adding `holiday_temp` and `weekend_temp`
interaction features fixes this: cold holidays now correctly predict higher consumption.

## LangChain agent

A conversational layer over the API. Ask in natural language instead of crafting JSON:

```
You: What's the forecast for next Friday?

Agent: Predicted consumption is 1,387 GWh (KNN). Using live SMARD data —
       yesterday's actual consumption was 1,352 GWh. Plausibility check passed.
```

The agent fetches real-time lag values from SMARD automatically, so predictions
use actual current grid data.

## Explainability (XAI)

SHAP and LIME explanations for the KNN model (`src/explain.py`), answering
*why* the model predicts what it does.

**SHAP beeswarm** — global feature importance across test predictions:

![SHAP summary](reports/shap_summary.png)

Weekend/weekday status and lag features have the strongest impact. Holidays are rare
but cause the largest single-prediction swings.

**SHAP waterfall** — breakdown of a single holiday prediction (New Year's Day):

![SHAP waterfall](reports/shap_waterfall.png)

Starting from the average prediction of 1,334 GWh, the holiday flag alone pushes it
down by 172 GWh. Weekend status and low lag values pull it further to a final
prediction of 1,053 GWh.

**LIME** — local surrogate explanation for the same prediction:

![LIME explanation](reports/lime_explanation.png)

Confirms the same story from a different method: holiday and weekend flags dominate,
with temperature being the only feature pushing consumption up (cold weather = more heating).

```bash
python3 src/explain.py
# Outputs → reports/shap_summary.png, reports/shap_waterfall.png, reports/lime_explanation.png
```

## How to run

```bash
# Train (auto-fetches latest SMARD data)
pip install -r requirements.txt
python3 src/train.py

# Start API
uvicorn src.api:app --reload

# Start LangChain agent (separate terminal)
pip install -r langchain_agent/requirements-langchain.txt
export ANTHROPIC_API_KEY=sk-ant-...
cd langchain_agent && python3 agent.py
```

## Stack

Python, Pandas, Scikit-learn, FastAPI, SHAP, LIME, LangChain, LangGraph, Streamlit,
Open-Meteo API, SMARD API, Docker

## Next steps

- LLM email parser to auto-extract special events from customer notifications
- RAG over energy documentation for contextual Q&A
- Drift monitoring and automated retraining via GitHub Actions