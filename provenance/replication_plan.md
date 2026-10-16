# June London replication, frozen before reading June outcomes

2026-10-16. September's primary review-credibility interaction is positive and significant. Its remaining checks are running. An older London snapshot was discovered from the provider's public dataset index, downloaded, and only its column header was inspected (same 90 columns). No June data rows or outcome relationships have been examined.

Source: https://data.insideairbnb.com/united-kingdom/england/london/2026-06-19/data/listings.csv.gz . The public get-the-data page's static query identifies June 19 as the preceding London release. Use this single release; do not replace it if results fail.

Apply the identical rating/review eligibility, £10–£2,000 bounds, rating contrasts (0.2 points, 10 versus 100 reviews), focus model and core controls. No train/validation selection: estimate the frozen model once on all eligible June listings, with preprocessing learned from that cohort without using outcomes beyond fixed price eligibility. This is temporal replication, not automatically independent because hosts can recur.

For a genuinely separate-host check, also estimate once on June listings whose host_id does not appear anywhere in the full September raw snapshot, including September listings without prices. Same June-fitted preprocessing. Report its sample size and precision; do not pretend overlap is eliminated merely by using a different month. Require both positive components and composite p*21<.05 for a successful independent-host replication; report failures.

June all-eligible robustness: squared continuous controls, equal-host total weights, borough-cluster uncertainty, £20–£1,000 prices, host fixed effects, 1-km grid fixed effects, add rating subscores, entire homes only and private rooms only. Use the same code and specification as September. Require all component directions and ordinary 95% intervals to meet the hypothesis to call temporal replication robust. Report all checks and 21-test adjusted p-values, including failures. Cross-snapshot comparisons do not identify a causal effect or estimate how any individual listing changed prices.

Save both raw dataset hashes and schema identity; source metadata; exact June eligible membership and September-host overlap flag; all estimates and tests; and independent contrast/cohort verification. No causal or 'proved mechanism' language.
