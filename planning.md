# Planning: Classifying Discourse Type in r/askphilosophy

## 1. Community

**Chosen community:** r/askphilosophy, labeling individual comments (answers), not submissions.

r/askphilosophy is a Q&A community: users post a question, and others answer. I chose it for three reasons. First, the unit of data is naturally self-contained — an answer comment stands on its own without needing the surrounding thread to make sense, which makes it cleanly annotatable. Second, the discourse genuinely varies in kind, not just quality: the same question will attract a rigorous reasoned answer, a one-line opinion, and a "here's what Kant actually said" textual summary. Those are different modes of philosophical talk, and the community itself recognizes them (mods routinely remove "this is just your opinion" answers). Third, it's high-volume and public, so collecting 200+ examples is feasible.

Crucially, I am classifying the **form** of the discourse, not whether the philosophy is correct. Judging correctness would require domain expertise, would make two annotators disagree constantly, and would collapse back into a vague good/bad taxonomy. Form is observable and annotatable without a philosophy degree, which is what makes this a tractable classification task.

## 2. Labels

Three labels. I considered a fourth (`clarifying_question`) but dropped it: in a Q&A sub the answers are overwhelmingly declarative, so that class would have captured too few examples to learn from, and the genuine clarifying questions are usually short top-level questions rather than answer comments.

### `argument`

The comment advances a claim and supports it with explicit reasoning — premises, distinctions, conditionals, or worked-through implications that visibly connect to the conclusion.

**The test:** if you strip the rhetoric, actual supporting reasons remain. The argument need not be correct — only present and visible.

**Examples:**

- "It doesn't follow that determinism rules out moral responsibility. Compatibilists distinguish freedom-as-could-have-done-otherwise from freedom-as-acting-from-one's-own-reasons. If responsibility tracks the second kind, then a determined agent who acts on their own deliberation is still responsible, because the relevant control is present even when alternatives aren't."
- "Hume's point isn't that induction is false, it's that it can't be justified without circularity: any defense of induction (it's worked before) is itself an inductive inference, so it presupposes what it's trying to establish."

### `assertion`

The comment states a philosophical position, judgment, or opinion without supplying supporting reasons — it tells you what to think but not why.

May be confident, may even be correct. The defining feature is the **absence of visible reasoning**. Name-dropping a thinker or using jargon does not make it an argument if no reasons are actually given.

**Examples:**

- "Honestly, Continental philosophy is mostly obscurantism dressed up as profundity. Analytic philosophy is the only tradition doing real work."
- "Free will obviously exists, anyone who denies it is just playing word games. This is settled."

### `exegesis`

The comment primarily explains or interprets what a specific text, thinker, or position says, rather than arguing for a view of the commenter's own.

The center of gravity is reporting/clarifying someone else's view ("Kant's actual claim is…", "in the Republic, Plato distinguishes…"), even if a small evaluative aside appears.

**Examples:**

- "When Nietzsche says 'God is dead' he isn't making a metaphysical claim about God's existence. In The Gay Science it's a diagnosis of culture: the Christian moral framework that grounded European values has lost its authority, and we haven't reckoned with the consequences."
- "Quick clarification on terminology: for Aristotle 'eudaimonia' isn't a feeling of happiness. It's closer to 'flourishing' — living well across a whole life in accordance with virtue and reason, which is why he says you can't call someone eudaimon until their life is complete."

## 3. Hard Edge Cases

The dominant ambiguity is **argument vs. assertion**, specifically the jargon-dressed assertion. A comment name-drops a philosopher and uses technical vocabulary so it sounds reasoned, but offers no actual premises:

> "This is just warmed-over Nietzschean ressentiment — classic slave morality projecting weakness as virtue."

It feels like argument because of the terminology, but strip the jargon and only the claim remains; nothing supports it.

**Decision rule:** Remove the technical vocabulary and proper names and read what's left. If supporting reasons survive that would back the claim on their own, label `argument`. If only the bare claim survives — the terminology was decorative, signaling expertise rather than doing argumentative work — label `assertion`.

A second, rarer edge case is **exegesis vs. argument**: a comment that explains a thinker's view and defends it. The rule here is **center of gravity** — if the bulk of the comment is reporting what the thinker said and the defense is incidental, label `exegesis`; if the commenter clearly takes the view as their own and the textual report is just setup for their argument, label `argument`. I will record the reasoning for any such call in the notes column.

## 4. Data Collection Plan

**Source & tooling:** Reddit's public listing for r/askphilosophy, top comments over the past year, collected with a script (`collect_reddit.py`) that pulls comment trees, drops AutoModerator/deleted/rules-bot comments, and filters comments outside 20–400 words. It writes a CSV with empty `label` and `notes` columns. This keeps collection reproducible and documented.

**Volume & target distribution:** Collect ~250 comments, label the first 200 that pass review (the buffer means edge cases I discard don't drop me below threshold). I do not expect a balanced natural distribution — I anticipate roughly `assertion` and `argument` as the larger classes and `exegesis` as the minority, plausibly something like 90 / 80 / 30 before any rebalancing.

**Underrepresentation contingency:** If any label has fewer than ~40 examples after the first 200 (likely `exegesis`), I will do targeted collection rather than relabeling: query threads where that class concentrates (e.g. "what did [thinker] mean by…" questions surface exegesis), and pull additional comments until the minority class reaches at least 40. I will report the final per-class counts and note that minority examples were targeted, since that affects how I read the evaluation. I will keep a held-out test set sampled **before** any targeted top-up, so the contingency can't leak into evaluation.

## 5. Evaluation Metrics

Accuracy alone is misleading here because the classes are imbalanced — a model that always predicts the majority class could score ~45% accuracy while being useless on `exegesis`. So:

- **Macro-averaged F1 (primary metric).** Averages F1 equally across the three classes regardless of size, so the minority `exegesis` class counts as much as the majority ones. This is the headline number I optimize.
- **Per-class precision and recall.** I need to know where the model fails, not just that it does. In particular, recall on `exegesis` (am I catching the rare class?) and precision on `assertion` (see success criteria — false "assertion" flags are the costly error for a real tool).
- **Confusion matrix.** The whole project hinges on the argument↔assertion boundary. The confusion matrix tells me whether errors are concentrated there (expected and tolerable — it's the genuinely hard distinction) or scattered randomly (a sign the model learned nothing real).
- **Baselines for context.** A majority-class baseline and a simple keyword/bag-of-words baseline. My model has to clear both by a meaningful margin to justify itself.
- **Inter-annotator / annotation-quality check.** Since I'm the sole annotator, I will re-label a random 30 examples a week after the first pass and compute my own agreement (Cohen's kappa). If my self-agreement is below ~0.7, my label definitions are too fuzzy and no model trained on them can be trusted — this gates the whole evaluation.

## 6. Definition of Success

Concrete, checkable thresholds:

- **Primary:** Macro-F1 ≥ 0.70 on a held-out test set. (Below ~0.60 the model isn't reliably distinguishing the classes; 0.70 is where per-class behavior becomes usable.)
- **No weak class:** Recall ≥ 0.60 on every individual class, including `exegesis`. A high macro-F1 that hides a 0.3-recall class is a failure.
- **Error shape:** The dominant off-diagonal error must be argument↔assertion. If `exegesis` is being confused with either at a comparable rate, the model hasn't learned the easy distinction and isn't ready.
- **Beats baselines:** At least +0.15 macro-F1 over the majority-class baseline.

**"Good enough for deployment" in a real community tool:** The realistic use is a triage assist that flags likely low-effort `assertion` answers for moderator review (not auto-removal). For that, the binding constraint is **precision on `assertion` ≥ 0.80** — a tool that wrongly flags genuine arguments as low-effort would frustrate good contributors and get switched off. I'd accept somewhat lower recall (it's fine to miss some) in exchange for high precision, because the cost of a false flag is much higher than the cost of a miss in a human-in-the-loop tool.

**Are these criteria objective?** Yes — each is a numeric threshold on a held-out set (macro-F1 ≥ 0.70, every-class recall ≥ 0.60, assertion precision ≥ 0.80, +0.15 over baseline) plus one structural check on the confusion matrix. At the end I can state pass/fail on each without judgment calls.

## 7. AI Tool Plan

This is an annotation/modeling project, not an implementation project, so AI tools help in three specific places rather than generating code.

### Label stress-testing — will do, up front

Before annotating, I will give an LLM my three label definitions plus the edge-case rules and ask it to generate 8–10 comments deliberately placed on the argument/assertion boundary (and a couple on the exegesis/argument boundary). I'll try to classify each using only my written rules. Any example I can't assign cleanly exposes a gap in my definitions, and I'll tighten the wording before touching the 200 examples. This is the highest-leverage AI use in the project.

### Annotation assistance — will use, with disclosure tracking

I will use an LLM to pre-label the dataset (zero-/few-shot with my definitions in the prompt), then review and correct every pre-label myself — the model proposes, I decide. I will add a `pre_label` column (the model's suggestion) and a boolean `pre_labeled` column to every row, so the final `label` (my reviewed call) is always distinguishable from the suggestion. This lets me (a) measure model–human agreement as a sanity check and (b) disclose exactly which rows were AI-assisted in my AI-usage write-up. I will not accept a pre-label without reading the comment myself.

### Failure analysis — will do, with manual verification

After evaluation, I'll export the misclassified test examples (text, true label, predicted label) and ask an LLM to cluster them and propose recurring failure patterns (e.g. "sarcastic assertions get read as arguments," "short exegesis gets read as assertion"). I will treat these as hypotheses only — for each proposed pattern I'll pull the specific examples it's based on and confirm the pattern actually holds before it goes in my write-up, since LLMs readily invent plausible-sounding patterns that don't survive inspection.
