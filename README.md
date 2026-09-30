# Learning to Summarize from Human Feedback (RLHF Pipeline Prototype)

A hands-on prototype of the first stages of an RLHF-style summarization pipeline, inspired by *Learning to Summarize from Human Feedback* (Stiennon et al., 2020). It fine-tunes a summarizer, generates candidate summaries, builds a preference dataset, and trains a **BERT reward model** to predict which summary is preferred.

> **Scope:** this is a small-scale learning project (coursework for Machine Learning and Pattern Recognition). Preferences are **simulated**, not collected from humans, and the final PPO stage is left as future work. See [Limitations](#limitations).

## Pipeline

**Phase 1 — Summarizer and candidate generation**
- Fine-tune `facebook/bart-large-cnn` on a 1,000-article subset of CNN/DailyMail (1 epoch, batch size 2)
- Generate two candidate summaries per article: **greedy** and **beam search** (4 beams)
- Simulate preferences by picking the candidate with the higher ROUGE-L against the reference summary

**Phase 2 — Reward model**
- Convert preferences into labeled pairs (`article [SEP] summary` → preferred = 1, not preferred = 0)
- Fine-tune `bert-base-uncased` as a binary classifier (3 epochs, learning rate 2e-5, 80/20 split)
- Evaluate with accuracy, precision, recall, F1, a confusion matrix and a classification report

**Phase 3 — Policy optimization (future work)**
- PPO with the reward model via Hugging Face `trl` (imports set up, not yet implemented)

## Results

**BART summarizer** (200 validation articles):

| ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|---|---|---|---|
| 0.329 | 0.137 | 0.236 | 0.304 |

**BERT reward model** (80 held-out pairs): accuracy 0.34, F1 0.18 for the "preferred" class.

The reward model performs at or below chance. That is expected given the setup: the two candidates were often identical, so the simulated labels carry little learnable signal, and the training set is tiny (about 320 pairs).

## Limitations

- **Simulated preferences.** ROUGE-L stands in for human judgment, and greedy and beam outputs frequently coincide, so "preferred" is almost always greedy
- **Small scale.** 1,000 training articles, 200 validation articles, 1 epoch for BART
- **Input truncation in candidate generation.** The candidate-generation helper truncates the article to 64 tokens, which limits summary quality
- **No PPO stage yet.**

## Next steps

- Use real human preference data (e.g. the OpenAI TL;DR comparisons dataset) instead of simulated labels
- Generate more diverse candidates (sampling, different temperatures, multiple models)
- Train the reward model with a pairwise ranking loss
- Complete the PPO fine-tuning stage with `trl`

## Repository structure

```
learning-to-summarize-from-human-feedback/
├── project.ipynb                                        # full pipeline
├── Machine Learning and Pattern Recognition Coursework paper.pdf
├── requirements.txt
└── README.md
```

## Getting started

```bash
git clone https://github.com/ishaan175pathak/learning-to-summarize-from-human-feedback.git
cd learning-to-summarize-from-human-feedback
pip install -r requirements.txt
jupyter notebook project.ipynb
```

A CUDA GPU is required as written (the notebook calls `.cuda()`). The CNN/DailyMail dataset downloads automatically through Hugging Face `datasets`.

## Tech stack

Python · PyTorch · Hugging Face Transformers, Datasets, TRL · BART · BERT · scikit-learn · ROUGE (`evaluate`) · Pandas · Matplotlib

## Author

**Ishaan Pathak** · [GitHub](https://github.com/ishaan175pathak)
