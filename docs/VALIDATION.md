# Validation record

Local verification on September 14, 2026, Windows, Python 3.12. The exact package versions are in `requirements-reference.txt` and the generated manifest.

- Engineering tests check nominal unit conversion, RF-fault direction, the sensor-bias negative control, recipe bounds, frozen statistical limits, hand-calculated SPC/Cp/OEE/MTBF/MTTR values, event accounting, same-seed reproduction, censored maintenance labels, independent matching confirmation and search budgets.
- Streamlit AppTest exercises all six analysis sections, changes an RF slider and checks that the model response changes, and opens the sensor-bias scenario.
- Browser verification confirmed that the dashboard renders, the navigation changes views, Plotly charts and tables are present, and the local browser error collection was empty. Screenshots are included in `reports/reference/`.
- A full seeded reference run generated 4,800 fleet wafers, 30 trials per SPC scenario, 14 maintenance campaigns and three paired recipe-search seeds.
- `scripts/verify_report.py` checks source/CSV hashes and independently recomputes event totals, reliability/OEE, test predictive scores, split separation, tool-matching spans and optimizer best-so-far histories.

The GitHub Actions workflow is supplied for Linux and Windows. It has not been run on GitHub during this local build; local passing checks do not imply a remote CI result. No physical etch validation was performed.
