# Classifying Discourse Type in r/askphilosophy

A 3-way text classifier that labels r/askphilosophy comments by the **form** of their
discourse — not whether the philosophy is correct. The project compares a zero-shot LLM
baseline against a fine-tuned DistilBERT model on a hand-labeled dataset of real comments.

**Headline result:** the fine-tuned model **underperformed** the zero-shot baseline
(0.765 vs. 0.882 accuracy, a 0.118 regression). The bulk of this README explains *why*,
because the failure is the interesting part.

---

## 1. Community and task

**Community:** [r/askphilosophy](https://reddit.com/r/askphilosophy), a moderated Q&A
subreddit where users answer philosophical questions. I classify individual **comments**
(answers), which are self-contained and vary widely in kind.

**Why this community:** the discourse varies in *form*, not just quality — the same
question attracts a tight reasoned answer, a bare opinion, and a "here's what the thinker
actually said" summary. Critically, I label the **form** of the discourse rather than its
correctness, so annotation doesn't require subject-matter expertise and the boundaries stay
stateable.

## 2. Labels

| Label | Definition |
|---|---|
| `argument` | Advances a claim and supports it with explicit reasoning — premises, distinctions, or worked-through implications (structure is visible: X, because Y, therefore Z). |
| `assertion` | States a position or opinion without supporting reasons. May be confident or even correct; name-dropping a thinker or using jargon does **not** count as an argument if no reasons are given. |
| `exegesis` | Primarily explains or interprets what a specific text or thinker *says*, rather than arguing for a view of one's own. |

I considered a fourth label (`clarifying_question`) and dropped it: in a Q&A sub the
answers are overwhelmingly declarative, so that class would have been starved.

**Hardest boundary (anticipated and confirmed):** `argument` vs. `assertion`, especially
the *jargon-dressed assertion* — a comment that name-drops a philosopher and uses technical
vocabulary but offers no actual premises. Decision rule: strip the terminology; if
supporting reasons remain it's `argument`, if only the bare claim remains it's `assertion`.

## 3. Data

- **Source:** real public comments collected from dozens of r/askphilosophy threads.
- **Size:** 222 labeled comments in a single CSV (`dataset.csv`), columns `text, label, notes`.
- **Split:** 70 / 15 / 15 train/validation/test, handled automatically by the notebook
  (test set = 34 comments).
- **Label distribution (full set):**

| Label | Count | Share |
|---|---|---|
| argument | 145 | 65.3% |
| exegesis | 53 | 23.9% |
| assertion | 24 | 10.8% |

This imbalance — **argument is ~6× more common than assertion** — turns out to be the
central driver of the fine-tuned model's failure (Section 6).

## 4. Methods

**Baseline — zero-shot LLM (Groq).** A general LLM is prompted with the label definitions
and one example per label, and instructed to output only the label name. No training. This
is a legitimate baseline: it measures how hard the task is for a capable general model with
zero in-domain training. 0% of responses were unparseable.

**Fine-tuned model — `distilbert-base-uncased`.** Fine-tuned on the training split on a
Colab T4 GPU.

| Hyperparameter | Value | Note |
|---|---|---|
| epochs | 10 | raised from default 3 (see Section 6) |
| learning rate | 3e-5 | raised from default 2e-5 |
| batch size | 16 | default |

## 5. Results

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Zero-shot baseline (Groq) | **0.882** | **0.86** |
| Fine-tuned DistilBERT | 0.765 | 0.50 |
| **Difference** | **-0.118** | **-0.36** |

### Per-class F1

| Label | Baseline F1 | Fine-tuned F1 |
|---|---|---|
| argument | 0.90 | 0.85 |
| assertion | 0.73 | **0.00** |
| exegesis | 0.93 | 0.67 |

### Confusion matrix — fine-tuned model (rows = true, columns = predicted)

| true \ pred | argument | assertion | exegesis |
|---|---|---|---|
| **argument** | 22 | 0 | 0 |
| **assertion** | 4 | 0 | 0 |
| **exegesis** | 4 | 0 | 4 |

The matrix tells the whole story: **every error is a minority class collapsing into
`argument`.** The model never predicts `assertion` at all, and catches only half of
`exegesis`. It achieves perfect `argument` recall (22/22) only by over-predicting it.

## 6. Analysis — why fine-tuning failed

### Two compounding problems, both visible in the training curve

The validation metrics by epoch (10-epoch run, lr 3e-5):

| Epoch | Train loss | Val loss | Val acc |
|---|---|---|---|
| 1 | 1.058 | 1.029 | 0.667 |
| 2 | 0.989 | 0.923 | 0.667 |
| 3 | 0.883 | 0.803 | 0.667 |
| 4 | 0.783 | 0.741 | 0.667 |
| 5 | 0.624 | 0.639 | 0.758 |
| 6 | 0.424 | 0.642 | **0.788** |
| 7 | 0.305 | 0.700 | 0.727 |
| 8 | 0.211 | 0.673 | 0.727 |
| 9 | 0.169 | 0.706 | 0.727 |
| 10 | 0.160 | 0.705 | 0.727 |

**Problem 1 — majority-class collapse (epochs 1–4).** Validation accuracy sits at exactly
**0.667** for the first four epochs — which is precisely the majority class's share of the
data. The model was doing nothing but predicting `argument`. It only "breaks through" at
epoch 5 (0.758) and starts using the other classes. With `argument` at 65% of training data,
predicting it by default is simply the lowest-loss strategy until the model is forced past it.

**Problem 2 — overfitting (epochs 7–10).** After peaking at epoch 6 (val acc **0.788**, val
loss 0.642), the curves diverge: training loss keeps falling (0.42 → 0.16 — the model is
**memorizing** the 155 training examples) while validation loss *rises* (0.64 → 0.70) and
validation accuracy slips back to 0.727. The final epoch-10 checkpoint — the one evaluated on
the test set at 0.765 — is therefore **past the model's optimal point.** Performance peaked
around epoch 6 and then degraded.

Both problems share one root cause: **the dataset is too small and too imbalanced.** 155
training examples (only ~17 of them `assertion`) gives the model little incentive to learn
the rare classes, and little material to learn from before it starts memorizing. The test-set
confusion matrix confirms the imbalance half — all 8 errors are minority→`argument`, and
`assertion` is never predicted at all.

A consequence worth stating: had I used early stopping (kept the best-validation checkpoint at
epoch 6 instead of the final epoch), the fine-tuned model would likely have scored slightly
higher — but still well short of the 0.882 baseline. The overall conclusion is unchanged.

### Three wrong predictions analyzed

1. **assertion → argument (confidence 0.63):** *"Well in that case, yes it's obvious why we
   can't envision that society. Because it's very unrealistic."* This is a bare assertion,
   but it literally contains the word **"because"** — a surface marker of reasoning. The
   model keyed on the connective rather than checking whether real support follows. A clear
   case of the model using surface features instead of discourse structure.

2. **assertion → argument (confidence 0.85):** *"I could point you to obliquely relevant
   political philosophy, normative ethics, and applied ethics of self-defense... but I can't
   point you to anything that directly addresses these specific problems."* No argument is
   actually made, but the comment is long, hedged, and dense with technical vocabulary. The
   model appears to read **sophistication as argumentation** — exactly the jargon-dressed
   trap my taxonomy warned about, now committed by the model instead of an annotator.

3. **exegesis → argument (confidence 0.88):** *"In Beyond Good and Evil the word abyss is
   used in several ways. Fighting with a monster contrasts gazing into an abyss, where the
   former suggests activity with others and the latter activity by yourself..."* This is
   textual interpretation, but its **contrastive, analytical sentence structure** ("the
   former... the latter...") resembles the connective tissue of an argument. The model
   confused analytical *prose style* with the act of arguing.

The common thread: the model latches onto **lexical/structural surface cues** (because,
technical vocabulary, contrastive connectives) rather than the underlying question — *is a
claim actually being supported, or merely stated/attributed?* That deeper distinction is
what separates my three labels, and 222 imbalanced examples were not enough for DistilBERT
to learn it over the easier surface heuristics.

### Why the baseline was so hard to beat

The zero-shot LLM scored 0.882 / macro-F1 0.86 — already above the macro-F1 ≥ 0.70 success
threshold I set in planning. A large pretrained model reasons about the *definitions* I
gave it, so it handles the rare classes without needing balanced training data (its
`assertion` recall was actually 1.00). DistilBERT, by contrast, only sees my 155 training
examples and inherits their imbalance. Beating a strong general LLM on a small, skewed
dataset is genuinely hard, and the milestone notes this is a legitimate, reportable outcome.

## 7. Limitations and shortcomings

- **Severe class imbalance (65/24/11).** The single biggest factor. The model had little
  incentive to predict `assertion`, and with only ~17 assertion examples in training it
  never learned the class — fine-tuned `assertion` F1 is 0.00.
- **Tiny test set for rare classes.** The test set has only **4 assertion** and **8
  exegesis** examples. These per-class numbers are extremely noisy; one comment flipping
  swings F1 substantially, so the rare-class metrics should be read as directional, not
  precise.
- **Small overall dataset (222).** Below what transformer fine-tuning typically wants for a
  subtle, definition-driven distinction.
- **Surface-feature reliance.** As the error analysis shows, the model leans on lexical cues
  rather than discourse structure — the hard part of the task.
- **Label provenance.** Labels were applied with an LLM using my planning.md definitions
  (see AI usage), which introduces the model's own biases into the ground truth; a fully
  manual pass would be stronger.

## 8. What I would do next (not implemented — out of scope for this lab)

- **Class-weighted loss or oversampling** to counteract the imbalance — the targeted fix for
  the diagnosed majority-class collapse. This is the first thing I'd try and likely the
  highest-impact change.
- **Collect more minority-class data**, especially `assertion`, to balance the classes
  rather than reweighting.
- **A larger held-out test set** so the rare-class metrics are trustworthy.

I deliberately stopped at diagnosis rather than implementing these, since the goal here was
to analyze the result, not to chase a better score.

## 9. AI usage disclosure

- **Data collection:** an AI assistant helped write the Python script used to pull and clean
  comments.
- **Annotation:** labels were applied by an LLM following my `planning.md` definitions, then
  intended for human review. This is disclosed because it affects ground-truth quality.
- **Baseline model:** the zero-shot baseline itself is an LLM (via Groq).
- **Debugging:** an AI assistant helped diagnose the notebook errors (label-map mismatch,
  Colab secrets) and interpret the training results.

## 10. Repository contents

| File | Description |
|---|---|
| `dataset.csv` | 222 hand-labeled comments (`text, label, notes`) |
| `planning.md` | Community, label definitions, edge cases, eval plan, AI tool plan |
| `evaluation_results.json` | Baseline vs. fine-tuned accuracy + label map |
| `confusion_matrix.png` | Fine-tuned confusion matrix (visual; the table in Section 5 is the canonical version) |
| `README.md` | This file |

### Reproduce

1. Open the Colab notebook; set runtime to **T4 GPU**.
2. Run Section 1 (define label map, upload `dataset.csv`) and Section 2 (split + tokenize).
3. Run Section 5 (add Groq API key + classification prompt) for the zero-shot baseline.
4. Run Section 3 (fine-tune), Section 4 (evaluate), Section 6 (compare + export).
