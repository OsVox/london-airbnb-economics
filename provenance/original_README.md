# Group project: predicting short-term rental prices in London

**Replicated economic finding, October 16:** [Review quantity complements review quality in asking prices](review-credibility-regression/report.html). The same rating improvement carries a larger price premium when backed by more reviews. The OLS interaction is supported in September and replicated in the June London snapshot, including hosts absent from September. The report includes all checks, a qualification for September's smaller within-host holdout test, and the distinction between a replicated association and proof of a causal information mechanism. The preceding two [group-travel interaction hypotheses](group-travel-regression/results/tests.csv) were not supported.

**New ordinary-regression screen, October 16:** [Bathroom privacy, distance-gradient curvature and bathroom congestion](privacy-access-regression/README.md). The private-bathroom premium exceeds the workspace premium consistently across all three host groups and all four robustness checks. The distance gradient flattens farther from the centre, with a qualification under the strictest spatial uncertainty correction. The bathroom-congestion interaction is unsupported. This screen uses no DDML and reports all three proposals as exploratory associations.

**DML interaction screen, October 16:** [Workspace × minimum stay, self-check-in × room type, and Superhost × centrality](heterogeneity-dml/README.md) were tested using host-grouped cross-fitting with XGBoost, repeated splits, linear nuisance models and OLS comparisons. None of these three differential-premium hypotheses is supported. This does not re-estimate the bedroom, capacity or minimum-stay coefficients below with DML; those earlier findings remain regression associations.

**Second economic screen, October 16:** [Six new hypotheses](economic-hypotheses-2/README.md) found consistent adjusted associations for lower price per guest-capacity unit, a separate-bedroom premium holding capacity/beds fixed, and lower nightly prices with longer minimum stays. Three amenity/quality interactions were not reliably supported. The supported results survive the predefined split, multiplicity and robustness checks, but remain exploratory associations from this same snapshot.

**Follow-up tests, October 16:** [Three hypotheses about Superhost status, reviews and multi-listing hosts](hypotheses/README.md) were tested with the existing host-level splits. None is robustly supported. That follow-up reuses the pilot's already examined test sample; its results are explicitly exploratory. The original location pilot below retains its original evaluation history.

**Current project framing (updated after reviewing the supplied 2022 examples):** predict nightly asking prices from listing, host, amenity, booking-date and location information; compare simple models with random forest and XGBoost; then interpret the performance differences and prediction errors. The central-location premium is one supported interpretation result within this broader prediction project. A positive hypothesis test or an original academic finding is not a project entry requirement.

## Why this is an economic question

Guests pay for access to employment, transport, attractions and other desirable urban activities. A central location can therefore command a price premium. The competing explanation is that central properties are larger, better equipped or otherwise higher quality. The analysis controls for observed property/host/amenity differences and booking-date context, while acknowledging that unmeasured quality remains.

The main research question is: **How accurately can observable listing characteristics predict nightly asking prices, and what explains differences in predictive performance?** The project has an established economic setting: prices reflect space, quality, location and market conditions. Its contribution for the course is careful data preparation, honest validation, model comparison and interpretation on recent observations. The supplied earlier projects are useful examples, not evidence of their grades or additional mandatory rules.

## Data verified in actual files

Source: [Inside Airbnb](https://insideairbnb.com/get-the-data/), [exact London detailed file](https://data.insideairbnb.com/united-kingdom/england/london/2026-09-17/data/listings.csv.gz), CC BY 4.0. Anonymous free download tested, compressed size 48.3 MB. Attribution: Inside Airbnb, London, September 2026 release, accessed October 16, 2026. [Source/provenance notes](SOURCES.md).

| Audit item | Verified result |
|---|---:|
| Raw listings / raw columns | 91,954 / 90 |
| Unique raw hosts | 51,348 |
| Duplicate listing IDs | 0 |
| Missing quoted price | 32,049 (34.85%) |
| Positive prices | 59,905 |
| Main usable sample, £10–£2,000 | 59,635 listings / 28,908 hosts |
| Train | 36,331 listings / 17,344 hosts |
| Validation | 11,400 listings / 5,782 hosts |
| Final holdout | 11,904 listings / 5,782 hosts |

The release is labelled September 17; actual row scrape dates are September 18–23, 2026. These are observed advertisements collected then, even where the quoted stay occurs later. They are not later realised transactions. Each listing enters once. Host groups, not rows, were randomly split with fixed seed 20261016, so a host cannot occur in both training and test.

**Do not treat the 90-column headline as predictor richness.** Fifteen original columns are entirely empty; some others are IDs, text descriptions, repeated summaries or direct copies of the target. After transparent parsing, the usable model inputs include 21 varying property/host/review measures, five booking-context transformations, 31 semantic amenities derived from ONE amenities-list column, and a four-input location group. The last group includes two coordinates, a neighbourhood label and a distance derived from the coordinates; it is not four independent raw measurements. Neither one-hot category levels nor the 31 extracted amenities are counted as 31 additional raw columns. Rich information remains useful, but the supplied course examples do not establish a minimum number of columns. The substantial usable sample and interpretable variables justify this choice without inflating feature counts.

Physical/host information includes room/property type, capacity, bedrooms, beds, bathroom count/sharing, hosting tenure, superhost/verification status and host listing count. Review measures describe rating/accuracy/cleanliness/check-in/communication and review volume. Amenities include kitchen, washer, heating, air conditioning, workspace, parking, lift and outdoor space. The full declaration and variation checks are in results/feature_manifest.csv. Bedrooms and beds have some missing values; numeric bathrooms is entirely empty, so the number is parsed from bathrooms_text. Empty response/booking fields contribute no useful variation.

The new raw quote records explicitly identify GBP, which resolves the misleading dollar symbols in the CSV. In 59,761 rows a valid nightly quote could be parsed; every one matched the price within rounding. Quotes use different check-in dates and stay lengths, so these contexts are included in all models. Price/total-price/raw-quote copies, estimated revenue/occupancy, review_scores_value and review_scores_location are excluded from predictors.

## Frozen hypothesis and pilot

The [analysis plan](analysis_plan.md) was written before downloading/inspecting prices. Source-schema clarifications and a computational change from sparse to dense tree input were recorded before score inspection. The interrupted initial runs did not influence target, cutpoints or tuning choices. Preprocessing and category learning used training rows only. Validation selected Ridge/XGBoost settings; the final holdout was not used for tuning.

P = property, host, amenity and booking-context inputs without geographic information. PD = P plus centre distance. PL = P plus coordinates, neighbourhood and centre distance. The fixed centre is 51.5074 N, −0.1278 E; a single centre is an intentionally simple approximation. No neural networks, complex spatial models or causal-ML methods were used.

### Interpretation check: the central-location premium

An ordinary log-price regression on **training data only**, controlling for P, estimates a distance coefficient of **−0.0392 per km**. A 5 km greater distance is associated with **17.8% lower asking prices**, with a host-cluster 95% interval of **16.9%–18.7% lower**. This is an adjusted association; it is not what would happen if the same apartment could be moved.

The descriptive training medians point in the same direction: whole-home listings within 3 km have median asking price £329 versus £199 at 6–10 km and £171 beyond 15 km. Those raw medians do not adjust for property differences. Private-room medians likewise decline from £109 to £78.32 and £63. Shared/hotel samples are small and not the basis for the claim.

### Independent ML evidence

Primary metric: mean absolute log-price error, which evaluates proportionate pricing errors. Lower is better. The GBP error is also shown for readability; predictions are obtained by exponentiating predicted log prices, without a separate bias correction. All numbers below use the same untouched host-level holdout.

| Model / inputs | Log-price MAE ↓ | GBP MAE ↓ | R² in log price ↑ |
|---|---:|---:|---:|
| Median baseline | 0.5794 | £116.10 | −0.004 |
| Ridge, no location | 0.3110 | £69.97 | 0.668 |
| Random forest, no location | 0.2966 | £66.72 | 0.691 |
| XGBoost, no location | 0.2826 | £64.32 | 0.722 |
| XGBoost, centre distance added | 0.2610 | £59.64 | 0.754 |
| XGBoost, full location added | 0.2511 | £57.10 | 0.767 |
| Ridge, full location added | 0.2813 | £62.97 | 0.712 |

**Adding centre distance lowers XGBoost's log-price error by 7.63%**, with paired host-cluster 95% interval **6.19%–9.16%**. **Adding full location lowers it by 11.16%**, interval **9.80%–12.48%**. Ridge also improves by 9.53%, interval 7.94%–11.02%, so this is not a benefit confined to one nonlinear algorithm. Both XGBoost comparisons clear the predeclared 5% practical-improvement threshold; that threshold is a project decision criterion, not an estimated profit gain.

A small tree was also fitted, and all 13 final model scores plus all 21 validation fits are saved in results/. These intervals resample hosts while holding the fitted models fixed. They do not include training/tuning uncertainty or a full model of shared spatial shocks. The random split tests unseen hosts within the same observed London market, not a future year or a new city/neighbourhood.

### Price-tail sensitivity

The price range was fixed before seeing prices. With all positive-price listings belonging to the same test hosts, the frozen models evaluate 11,956 listings (52 extra expensive listings). XGBoost log errors remain 0.2916 without location, 0.2697 with distance and 0.2597 with full location. Thus the direction survives removing the upper-price restriction; full-location gain is about 10.93%. GBP errors become larger (£84.57 → £77.10), as expected when expensive outliers return. No model was refitted or selected on this sensitivity. Hosts appearing only in the excluded tails had no original split assignment and are outside it.

## Limitations and project fit

- The target is a nightly asking quote; it is not realised rent, revenue, willingness to pay, occupancy or social welfare. Quote components and inclusion of fees need to be stated exactly; we do not claim it is a fully comparable all-in transaction price.
- Missing prices remove about 35% of raw listings and may select more active/bookable properties. We do not generalise to every Airbnb listing or London's entire housing stock.
- Coordinates are intentionally perturbed, reportedly by up to roughly 150 m. This is small compared with a 5 km gradient but affects fine location comparisons.
- Room counts and amenities cannot measure every aspect of interior quality. Demand, supply, neighbourhood appeal and host pricing choices are not separately identified. The centre-distance coefficient is not causal.
- The 90 raw columns contain substantially fewer independent usable structured fields. This is a declared compromise, not solved by counting dummy columns or IDs.
- No ongoing competition is involved. This public alternative is recommended because actual access and a supported simple relationship were verified.

The course exercise can now combine a clear economic motivation, missingness/data description, readable price-distance plots, median/linear/tree baselines, forest/XGBoost comparisons, a host-group validation experiment, feature-group comparisons, and an interpretable conclusion. Next steps are to read the selected prior paper fully, compare physical/location/amenity information on development data, and run a prespecified spatial-block robustness check while keeping the prediction target fixed. A saved descriptive error audit (results/interpretation_error_groups.csv) reports where the frozen XGBoost model performs poorly by room type and price band. These are interpretation diagnostics, not a new model-selection test. Do not tune against the consumed holdout or pretend that a follow-up check is a new untouched final test.

## Reproduce

From the parent econ-dataset-search directory, using its existing isolated environment:

```sh
.venv/bin/python airbnb-location/scripts/download.py
.venv/bin/python airbnb-location/scripts/profile.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python airbnb-location/scripts/pilot.py
.venv/bin/python airbnb-location/scripts/verify.py
.venv/bin/python airbnb-location/scripts/plot.py
```

Dependencies are frozen in ../requirements.lock.txt. The data URL and checksum are retained in logs/download.json; preserve the downloaded snapshot. Verification independently reproduced all model MAE/R² values from saved predictions within floating-point/CSV precision and checked sample identities, host-disjoint splits, forbidden predictors and the frozen-plan hash. The figure results/overview.png was visually inspected.

The original RECS and PNCP/labour artifacts were preserved. No PDF/deck, competition submission, publication, purchase or external message was created.
