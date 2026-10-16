# Review precision and the rating-price premium

Fixed 2026-10-16 before estimating this new interaction. The two group-travel interactions just tested failed. They remain part of the counted search family: 20 prior substantive hypotheses, 21 including this one. One new hypothesis only, all results reported, OLS only.

## Hypothesis

A higher overall guest rating is associated with a larger asking-price premium when supported by more reviews. Economic mechanism proposed: a high average based on many observations is a more credible signal of quality than the same average based on few observations. This mechanism is not itself identified causally. Review counts also reflect age, demand and price; variance of review-level ratings is not observed.

Focus model: log price = a*(rating-4.8) + b*(log(1+reviews)-log(11)) + c*(rating-4.8)*(log(1+reviews)-log(11)) + controls. Fix comparisons at 10 versus 100 reviews and a rating increase of 0.2 points (e.g. 4.7 to 4.9). The joint hypothesis requires a positive rating-price slope at 100 reviews, and a larger slope at 100 than 10 reviews (c>0). Use maximum of the two two-sided p-values for the composite test and require both expected directions. Compare coefficients directly, not significance in one subgroup versus another.

## Cohort and specification

Same London September 2026 snapshot and original labelled £10–£2,000 host-disjoint splits; entire homes and private rooms; overall rating observed in [3,5], number of reviews 1–500. Original splits have been examined before: explicitly exploratory. No new validation set is claimed.

Controls: existing observed layout, property/room type, amenities, host tenure/badge/verification, reported host portfolio, minimum stays, quote timing and stay length, coordinates, borough and centre distance. Capacity log, host listing count log(1+count), minimum nights log. Exclude review volume and overall rating from nuisance controls because they are focus terms. Exclude the other guest rating subscores from the primary model, because they are alternative evaluations of quality rather than independently observed physical quality; add them in a specified sensitivity. Keep reviews per month as a demand/age proxy, acknowledging it is not a causal pre-treatment control. No target copies, occupancy/revenue estimates, IDs or descriptions.

Train-fitted median imputation, missingness indicators, scaling and one-hot categories (min frequency 20). Fit identical OLS separately in train, validation and old holdout. Host-cluster covariance with t(df=hosts-1). Require composite p*21<.05 and predicted directions in training and old holdout, and predicted validation directions. Report ordinary 95% intervals, all primary components and review-specific premium contrasts. The finite correction does not eliminate selective hypothesis generation.

## Prespecified checks

Training and old holdout: squared continuous nuisance controls; equal total weight per host; borough-cluster uncertainty; £20–£1,000 price restriction; host fixed effects; 1-km grid fixed effects; add all available guest-rating subscores (excluding location/value); entire-home-only; private-room-only. The last two use the same training-fitted core preprocessing. Fixed effects retain groups with two or more observations and account for absorbed indicators in degrees of freedom. A failed ordinary joint direction/CI check qualifies robustness. No changing review comparison, rating contrast or clipping limits after results.

## Verification

Save source/plan/code hashes, membership, controls, full coefficients/covariance and all contrasts/tests. Check direct full-design OLS and independent contrast arithmetic. Report whether estimates are informative enough to be useful, not just p-values. If supported, seek one accessible older London snapshot from the same provider; freeze its replication before examining outcomes. External replication is only claimed if actually run. No assertion of causal proof or academic novelty.
