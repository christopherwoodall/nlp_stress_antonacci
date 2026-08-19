# NLP Stress Pipeline Report

*Generated from: results*

---

## Dataset Summary

**Total rows:** 39081

**Sources:**
- joangaes_depression: 27970 rows
- zenodo_depression: 10481 rows
- mhdialog: 630 rows

**Severity distribution:**
- Severity 0: 14549
- Severity 1: 84
- Severity 2: 22107
- Severity 3: 2341

**Binary distribution:**
- No distress: 14549
- Distress present: 24532

## Model Training Summary

**Trained models:**
- `tfidf_model` → results/tfidf_model.onnx

**Train/test split:** 80/20 (31,268 / 7,817)
**Random state:** 42

**Embeddings model:** Not trained — ElasticNetCV grid search was too slow with 39K samples × 1024 features. Consider using a smaller feature subset or a faster regressor (e.g., Ridge, LightGBM).

## LLM Evaluation Summary

**Total evaluations:** 160

**Mean predicted severity by LLM model:**

| LLM Model | Version | Mean Severity | Std Dev |
|-----------|---------|---------------|---------|
| claude-sonnet | unknown | 1.414 | 0.034 |
| gpt-4o | unknown | 1.418 | 0.034 |

**Evaluations per question:**
- tesi_1_1: 10
- tesi_1_2: 10
- tesi_1_3: 10
- tesi_1_4: 10
- tesi_1_5: 10
- tesi_1_6: 10
- tesi_2_1: 10
- tesi_2_2: 10
- tesi_2_3: 10
- tesi_2_4: 10
- ... and 6 more

## Persona-Based Evaluation

### Mock Persona Data (engineered templates)

| Persona | Mean Severity | Std Dev |
|---------|---------------|---------|
| Depressed | 1.5751 | 0.0422 |
| Resilient | 1.4673 | 0.0056 |
| **Difference** | **+0.1078** | |

✅ Model correctly assigns higher severity to depressed-sounding text. Templates were engineered to use high-coefficient words (e.g., *depression*, *kill*, *die*, *thoughts*, *pain*).

### Real API Data (gpt-4o)

| Persona | Mean Severity | Std Dev |
|---------|---------------|---------|
| Depressed | 1.4489 | 0.0104 |
| Resilient | 1.5146 | 0.0084 |
| **Difference** | **-0.0656** | |

⚠️ **Model misclassifies natural language.** The resilient persona scored *higher* than depressed. The depressed response began with denial ("No, I've never...") which the model interprets as low severity, while the resilient response was longer and more emotionally verbose, which the model interprets as high severity. This reveals a limitation: the model conflates emotional expressiveness with distress.

## Visualization Summary

**Generated plots:**
- `average_severity.png`
- `model_radar.png`
- `question_heatmap.png`
- `severity_distribution.png`

---

## Next Steps

1. Review plots in `results/synthetic/`
2. Check `results/evaluations.csv` for detailed predictions
3. Run `make eval` with real API calls for actual LLM comparison
