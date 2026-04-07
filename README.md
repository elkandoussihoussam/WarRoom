# "Who Am I, and Who Else Is Here?"

**Behavioral Differentiation Without Role Assignment in Multi-Agent LLM Systems**

> When multiple large language models interact in a shared conversation, do they develop differentiated social roles or converge toward uniform behavior?

## Overview

This repository contains the data, analysis code, and figures for a controlled experimental study of emergent behavioral differentiation in multi-LLM group conversations. We orchestrate simultaneous discussions among 7 heterogeneous LLMs across 12 experimental series (208 runs, 13,786 coded messages), finding that:

1. **Heterogeneous groups** exhibit significantly richer behavioral differentiation than homogeneous groups (cosine similarity 0.56 vs. 0.85; *p* < 10⁻⁵, *r* = 0.70)
2. **Compensatory responses** emerge spontaneously when an agent crashes
3. **Real model names** significantly increase behavioral convergence (cosine 0.56 → 0.77, *p* = 0.001)
4. **Removing prompt scaffolding** converges profiles to homogeneous-level similarity (*p* < 0.001)

These behaviors are absent when agents operate in isolation, confirming that behavioral diversity is driven by the interaction of architectural heterogeneity, group context, and prompt-level scaffolding.

## Repository Structure

```
.
├── README.md
├── generate_figures.py
├── requirements.txt
├── .gitignore
├── LICENSE
├── platform/
│   └── war-room-lab-isolated.html
├── data/
│   ├── coded_v4_agreed.csv
│   ├── coded_v4_gemini.csv
│   ├── coded_v4_sonnet.csv
│   └── human_coding_results.csv
└── figures/                    # generated, not tracked
```

## Data Description

### Coding CSVs (13,786 messages each)

| Column | Description |
|--------|-------------|
| `id` | Unique message identifier (`{series}-{run}\|{agent}\|R{round}`) |
| `series` | Experimental series (A, B, C, E, F, G, H, I, J, K1, K2, K3) |
| `file` | Source JSON transcript filename |
| `nickname` | Agent nickname in the conversation |
| `agent` | Underlying LLM model |
| `round` | Conversation round (0–10 or 0–20) |
| `is_phatic` | Phatic/greeting behavior flag |
| `is_meta` | Meta-commentary flag |
| `is_lead` | Leadership behavior flag |
| `is_arch` | Architecture/technical discussion flag |
| `is_agree` | Agreement/validation flag |
| `is_comp` | Compensatory response flag (references crashed DeepSeek agent) |
| `comp_level` | Compensation depth: L1 (mention), L2 (takeover), L3 (redistribute) |
| `has_xref` | Cross-references other agents by name |
| `lang` | Detected language |
| `text` | Full message text |

- **`coded_v4_agreed.csv`**: Conservative intersection — a flag is True only when *both* judges agree
- **`coded_v4_gemini.csv`**: Gemini 3.1 Pro judge labels
- **`coded_v4_sonnet.csv`**: Claude Sonnet 4.6 judge labels

### Human Validation (609 messages)

Stratified random sample coded by a human annotator, with corresponding LLM judge labels for computing inter-rater agreement (mean κ = 0.73 vs. Gemini).

### Experimental Series

| Series | Description | Runs |
|--------|-------------|------|
| A | Baseline (heterogeneous, neutral names, French) | 21 |
| B | Homogeneous (8× LLaMA 3.3 70B) | 21 |
| C | Real model names | 21 |
| E | Shuffled agent order each round | 21 |
| F | Isolation (agents respond independently) | 21 |
| G | English language | 21 |
| H | Festival scenario | 15 |
| I | High temperature (0.95) | 11 |
| J | 20 rounds | 11 |
| K1 | Ablation: no peer list | 15 |
| K2 | Ablation: no identity | 15 |
| K3 | Ablation: empty system prompt | 15 |

## Reproducing Figures

```bash
pip install numpy matplotlib scipy
python generate_figures.py --data-dir data --output-dir figures
```

This generates all 11 figures in `figures/`. Runtime: ~30 seconds.

## Citation

```bibtex
@inproceedings{anonymous2026behavioral,
  title={``Who Am I, and Who Else Is Here?'' Behavioral Differentiation Without Role Assignment in Multi-Agent LLM Systems},
  author={Anonymous},
  year={2026}
}
```

## License

Data and code are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Raw JSON transcripts are available upon request.
