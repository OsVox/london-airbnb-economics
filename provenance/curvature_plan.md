# Post-result shape checks

2026-10-16, after the supported September and June estimates, including the successful June-only-host replication. These are explicitly post-result validity diagnostics, not additional independent confirmation.

The core model assumes linear rating and log-review main relationships. Review-count groups can have different rating distributions, so a misspecified rating curve could mimic an interaction. Add squared centered rating and squared centered log-review count to nuisance controls. Then repeat that curved-main-effects model restricting ratings to 4.5–5 to test whether low-rating observations drive the finding. Fix these two checks before estimating them. Use September training and old holdout, June full eligible sample and June-only hosts. Preprocessing and eligibility otherwise follow the original specifications.

Use the exact symmetric comparison rating 4.7 to 4.9 around 4.8. The rating-square term contributes zero to this contrast, while review-only terms cancel at fixed review count. Thus the saved focus contrast vectors remain exact for this comparison. The interaction difference remains directly testable. Report all eight fits, even if the joint claim fails. Do not replace the primary models or treat post-result diagnostics as preregistered.
