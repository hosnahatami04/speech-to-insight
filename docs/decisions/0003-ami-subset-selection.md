# 0003. AMI meeting subset under the CPU budget

## Context

The AMI corpus has a standard partition of 136 train, 18 dev and 16 test
scenario meetings. Running recognition, diarization and extraction over every
meeting on CPU is far beyond the project's time budget, and the project's value
is in measuring error propagation carefully, not in corpus scale. A subset is
needed, chosen without touching the test meetings during tuning.

## Decision

- Keep the official AMI partition (embedded as data in `data/ami_splits.py`,
  taken from the standard scenario-meeting lists) as the source of truth for
  which split a meeting belongs to.
- Evaluate on a fixed subset of the official dev and test splits, configured in
  `configs/data.yaml` (`ami.dev_meetings`, `ami.test_meetings`). The initial
  subset is three dev and three test meetings, one from each scenario family
  (ES, IS, TS), chosen so that each selected meeting has both a ground-truth
  RTTM and an abstractive summary with decision/action items.
- A meeting is only eligible if it has both an RTTM and a summary. Some dev
  meetings (for example the IB-series) have no abstractive summary and are
  excluded from extraction evaluation for that reason.
- Summary placeholder sentences (`*NA*`, `NA.`) mark an empty section and are
  dropped, so they never become reference decision or action items.

## Rejected options

- **All AMI meetings**: infeasible on CPU within the time budget.
- **A random subset**: would not guarantee scenario-family coverage or the
  presence of summaries, and would be harder to reason about.
- **Fetching the split lists at runtime** from a third-party repository: adds an
  undocumented network dependency and breaks offline reproducibility.

## Consequence

A small, reproducible, fully offline AMI subset that still spans the scenario
families and always has the references each evaluation needs. The subset size
can be increased later by editing the config, at a known CPU cost. Because the
meetings are acted scenario meetings, this limitation is recorded in the data
card and will be stated in Known issues.
