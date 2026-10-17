# London Airbnb: group project

Выполненный notebook и русская презентация исследуют прогноз ночной **цены предложения**, а также взаимодействие рейтинга и числа отзывов. OLS — обычная регрессия log(price), не logistic regression. Верхняя граница сложности — XGBRegressor.

## Открыть результаты

- `london_airbnb_group_project.ipynb` — расчёты, текст и видимые результаты выполнения.
- `london_airbnb_group_project.pptx` — редактируемые тексты, таблицы и графики; 18 основных слайдов + подробное приложение (все 136 коэффициентов и 40 проверок).
- `london_airbnb_group_project.pdf` — экспорт презентации.
- `speaker_notes.txt` — русские заметки; тот же текст встроен в PPTX.
- `requirements_mapping.csv` — где выполнено каждое требование.
- [План проверки проекта четырьмя участниками](docs/team-review-four-parts.docx) — задачи и ожидаемый результат для каждого блока.
- `results/visual_review.json` — результат просмотра всех 54 страниц PDF.
- `results/verification.json`, `results/completion_checks.json` — вычислительные проверки и проверка поставки.

## Сравнение разбиений

В разделе 6 notebook добавлен [график одного разбиения и GroupKFold](figures/split_vs_groupkfold.png) ([SVG](figures/split_vs_groupkfold.svg), [численные данные](results/split_vs_groupkfold.csv)). Он сравнивает исходную **validation** с тремя CV-частями при одинаковых настройках: Ridge α=10 и XGBoost depth=5. Оба способа разделяют данные по хозяевам. Средняя log MAE в CV выше на 0,55% и 0,99% соответственно; в CV также меньше обучающих объектов. Это проверка чувствительности к разбиению и объёму обучения, а не сравнение случайного разделения строк с разделением по хозяевам и не доказательство переобучения.

Перестроить график из сохранённых расчётов без переобучения моделей:

```sh
.venv/bin/python helpers/split_comparison.py
```

## Находка из SHAP: число ванных и тип жилья

В разделе **7.2.1** добавлен [кейс дополнительной ванной](docs/shap-bathroom-case.md) как расследование: сначала [зацепка — строка bathroom_count из общего SHAP-графика](figures/bathroom_shap_clue.png), затем [SHAP-взаимодействие](figures/bathroom_shap_interaction.png) и [ответ моделей при замене одной ванной на две](figures/bathroom_model_response.png). На исходном test XGBoost даёт +18,90% для целого жилья и −1,87% для отдельной комнаты; исходная OLS даёт +7,25% для обоих типов. Это разведочная находка о поведении моделей, а экономическое объяснение остаётся гипотезой. Код воспроизведения: `helpers/bathroom_case.py`; таблицы и методические сведения: `results/bathroom_*`.

## Основной результат и оговорки

XGBoost воспроизводит log MAE 0.2511 и GBP MAE £57.10 на исходном сентябрьском test (11,904 строки). Для Ridge GBP MAE £62.97. В отдельной OLS на рейтинговом старом test (9,514 строк) переход 4.7→4.9 связан с +5.18% при десяти отзывах и +11.18% при ста. Июньские проверки отдельно поддерживают направление, включая хозяев, отсутствующих во всём сентябрьском файле. Сентябрьский within-host test не проходит поправку на 21 гипотезу (p_adjusted≈0.317).

Гибкая matched XGBoost-модель показывает значительно более слабое различие: +8.33% против +8.96%. Включение взаимодействия в matched OLS улучшает test log MAE с 0.24885 до 0.24568 (около 1.28%). Значимость, точность и интерпретация модели отвечают на разные вопросы. Причинный механизм доверия к отзывам не установлен.

Оба снимка уже изучались. Это воспроизведение и заранее ограниченное расширение, не новая проверка на нетронутых данных. Июнь — более ранний снимок того же города, не прогноз будущего. Названия team members не выдуманы. Проект сохранён в приватном GitHub-репозитории; сдача преподавателю и репетиции не выполнялись.

## Reproduce the analytical notebook

Tested with Python 3.12.7, macOS arm64. Run **from this directory**. Dependencies are fully pinned in `requirements.lock.txt`. Raw data and the original split are included. No dependency on the parent project or chat is required for analysis.

Clone and run (Python 3.12 and uv required; XGBoost on macOS also needs an OpenMP runtime such as Homebrew libomp):

```sh
gh repo clone OsVox/london-airbnb-economics
cd london-airbnb-economics
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python requirements.lock.txt
.venv/bin/python helpers/download_data.py
.venv/bin/python helpers/verify.py
.venv/bin/python helpers/execute_notebook.py
.venv/bin/python helpers/verify.py
```

The first verification checks the included saved results. Executing the notebook then recomputes the models and inference checks. The notebook already contains executed outputs, so it can also be read without rerunning. Dependency pins were tested on macOS arm64; another operating system may require platform-specific package adjustments.

`execute_notebook.py` starts a new kernel with the invoking Python, executes every code cell, saves outputs, and shuts down the kernel. Thread counts are capped at four. The 40 OLS fits and all models are recomputed. Allow several minutes and roughly a few GB of RAM; exact time depends on hardware. It does not load an old report as analysis. `helpers/build_notebook.py` regenerates the source notebook and **clears its outputs**; only run it when editing, then execute again.

Read `analysis_plan.md` before changing the experiment. No fresh search across hypotheses, cutoffs or hyperparameter spaces is part of the reproduction.

## Data and checksums

Attribution: Inside Airbnb, London, CC BY 4.0, accessed 16 October 2026. One row is an advertised listing. The September release label differs from row scrape dates (18–23 September). Quote currency fields establish GBP. We use quote dates and stay length as context, never quote amounts as predictors.

- [September detailed listings](https://data.insideairbnb.com/united-kingdom/england/london/2026-09-17/data/listings.csv.gz)
- [June detailed listings](https://data.insideairbnb.com/united-kingdom/england/london/2026-06-19/data/listings.csv.gz)
- [Provider data directory](https://insideairbnb.com/get-the-data/)

| File | SHA256 |
|---|---|
| `data/london_2026-09-17.csv.gz` | `8017870734a9bffa1b424937372e5f02f8c4a114e5060534d7423be5fd718b4c` |
| `data/london_2026-06-19.csv.gz` | `f32fbf1bdd3a4fcdb58ffe901b4ac20d8614f77cd36f1e3bbd18523e6e2acabd` |
| `data/split.csv` | `bd474ede43770b629676e9048d25960ee41646a7e018f0b8c3872824d2b9f2e8` |

The download helper downloads only a missing snapshot, verifies its exact hash, and fails on changed content. `split.csv` is included; the notebook independently reconstructs it from the original host-split seed (20261016). Do not create a new split and describe it as untouched.

## Files and technical choices

- `helpers/features.py`, `ols_core.py`, `analyse.py`, `replicate.py`, `shape_*.py`: copied verified functions/specifications with only self-contained path adaptation.
- `helpers/analysis.py`: data audit, prediction, grouped CV, matched ablation, host bootstrap, permutation, exact TreeSHAP, response/support diagnostics, figures.
- `results/ols/`: all 40 fits, covariance matrices, memberships, full controls, components, contrasts and checks.
- `results/ols_full_training_coefficients.csv`: all 136 primary-training coefficients, standardized controls. Design rank 135: aliased-control coefficients are minimum-norm descriptions, not unique structural effects. Main unscaled focus inference is in `ols_focus_coefficients.csv`.
- `results/predictions_*.csv`, `matched_predictions.csv`, `model_scores.csv`, `development_cv.csv`: reproducible score inputs and outputs.
- `results/shap_*.csv`, `response_*.csv`, `hist_*.csv`, `location_medians.csv`: numerical plotting inputs. TreeSHAP contributions sum to model predictions. Tree-path frequencies from training define its background; no separate interventional background sample.
- `figures/`: actual computed charts; no generated imagery is used as evidence.
- `models/`: saved main XGBoost JSON and fitted preprocessor. Load joblib only from a trusted copy of this package.
- `provenance/reference_*`: historical comparison artifacts, clearly separated from newly calculated results. The notebook reruns and checks them; it does not substitute them for calculations.
- `provenance/original_files_sha256.json`: hashes of earlier project files at the start. Previous work was preserved.
- `provenance/course_pages_4-9.txt`: text read directly from Lecture 6.pdf. The five supplied 2022 presentations were inspected as examples; their scores were not independently validated.
- `provenance/literature_sources.json`: original source URLs and limits of the literature check.

September prediction models share train-fitted preprocessing and exactly the same full-location input group. Numeric imputation uses medians/missing indicators; categories are learned on train. Empty columns and one-hot levels are not counted as new independent fields. For inference, controls and cluster corrections match the original specification. June preprocessing is learned from eligible June covariates under the original replication plan; June regressions are separately estimated association checks, not train-model predictions.

All illustrative 95% coefficient intervals are ordinary cluster intervals, not simultaneous intervals corrected for 21 tests. Composite p-values use the maximum of two two-sided component p-values and Bonferroni×21. The correction does not undo adaptive hypothesis generation. Exact TreeSHAP values and partial model responses are neither significance tests nor causal estimates.

## Editable presentation source

The deck was authored with `@oai/artifact-tool` using the Codex bundled Node runtime. Charts contain editable literal data and embedded spreadsheet snapshots, rounded to ten significant digits for Excel compatibility; analytical CSVs retain the original precision. Tables and all text remain native slide objects. The PDF is exported from the finalized PPTX using LibreOffice Impress, except page 34: its two scatterplots are rendered by Artifact Tool and embedded as a high-resolution image, because LibreOffice drops some scatter series. The delivered PPTX retains the native editable charts. All other PDF pages preserve vector text and graphics. The direct Artifact Tool PDF exporter lacked its bundled Helvetica module; the LibreOffice fallback completed successfully. Native slide import, package integrity, font, geometry and chart-data validators are recorded in `.build/final-validation-v3.json`. This does not claim that Microsoft PowerPoint itself was opened.

Source: `.build/build_deck.mjs`, `.build/main_notes.json`, `helpers/export_slide_data.py`. This presentation renderer is a separate, environment-specific dependency; it is **not needed** to rerun the notebook. The editable PPTX and PDF are already delivered. On this Codex installation:

```sh
.venv/bin/python helpers/export_slide_data.py
/Users/andrejvorsin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node .build/build_deck.mjs
/Users/andrejvorsin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 helpers/export_pdf.py --soffice /Users/andrejvorsin/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/soffice
```

For a revised deck, change both the builder's `slides/final-v3.pptx` output and `.build/final-validation-v3.json` receipt to fresh version names first: the finalizer deliberately refuses to overwrite an existing finalized version. Its packaged validator paths are explicit in the builder. The local `.build/node_modules` symlink is not committed. To rebuild slides, create it pointing to the bundled Node modules and adapt the explicit runtime paths to your installation. Analytical reproduction does not need this renderer. Do not install unrelated replacement presentation libraries and assume equivalent output.

## Primary references

Gyódi, K., & Nawaro, Ł. (2021). *Determinants of Airbnb prices in European cities: A spatial econometrics approach*. Tourism Management, 86, 104319. https://doi.org/10.1016/j.tourman.2021.104319

Fradkin, A., Grewal, E., & Holtz, D. (2021). *Reciprocity and Unveiling in Two-Sided Reputation Systems: Evidence from an Experiment on Airbnb*. Marketing Science, 40(6), 1013–1029. https://doi.org/10.1287/mksc.2021.1311

Fradkin, A., & Holtz, D. (2021). *More Reviews May Not Help: Evidence from Incentivized First Reviews on Airbnb*. Author preprint. https://arxiv.org/abs/2112.09783

The literature review uses verified original abstracts/available method excerpts and does not claim a complete systematic review or academic novelty.

`helpers/verify_delivery.py` is the local delivery audit: in addition to analytical checks it expects PDF page renders and the original parent project for the preservation check. These temporary renders and the parent project are not needed for notebook reproduction and are not included in this repository.

The PDF export helper uses `pypdf` and `reportlab` from the bundled document runtime, independently of the analytical Python environment. Their versions are saved in `provenance/document_runtime.json`. The included `slides/slide-34-render.png` must be regenerated with the builder if that slide changes.


## Repository contents and attribution

The two compressed raw snapshots are included directly in `data/`; Git LFS is not required. Download dates, original URLs, CC BY 4.0 attribution and checksums are recorded above. Data licence: https://creativecommons.org/licenses/by/4.0/. Raw snapshots have not been modified. Derived datasets and analysis outputs are identified by their folders.

The repository keeps the completed deliverables and required reproduction files. Temporary slide builds, caches, duplicate presentation versions and the redundant full-project ZIP are excluded. `MANIFEST.sha256` records the initial committed file contents; after rerunning the notebook, generated outputs may intentionally differ. The original local project is preserved separately.

The earlier course and presentation excerpts in `provenance/` are reference material for internal coursework review; they are not a grading rubric or independently verified studies.
