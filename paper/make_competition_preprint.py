"""Generate the competition preprint as a polished, self-contained PDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    Flowable, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
    Table, TableStyle,
)


NAVY = colors.HexColor("#13243A")
INK = colors.HexColor("#18212B")
MUTED = colors.HexColor("#5B6875")
TEAL = colors.HexColor("#0B7A75")
PALE_TEAL = colors.HexColor("#E7F4F2")
ORANGE = colors.HexColor("#E67E22")
PALE_ORANGE = colors.HexColor("#FFF2E6")
LINE = colors.HexColor("#D7DEE5")
PAPER = colors.HexColor("#FAFBFC")
WHITE = colors.white


class BarChart(Flowable):
    def __init__(self, rows, width=165*mm, height=54*mm, lo=0.68, hi=0.95):
        super().__init__()
        self.rows, self.width, self.height, self.lo, self.hi = rows, width, height, lo, hi

    def draw(self):
        c = self.canv
        label_w, right = 53*mm, 14*mm
        chart_w = self.width - label_w - right
        row_h = self.height / max(len(self.rows), 1)
        for i, (label, value, color) in enumerate(self.rows):
            y = self.height - (i + 0.72) * row_h
            c.setFillColor(MUTED)
            c.setFont("Body", 7.3)
            c.drawString(0, y + 1.2*mm, label)
            c.setFillColor(colors.HexColor("#E8EDF1"))
            c.roundRect(label_w, y, chart_w, 4.5*mm, 2.25*mm, fill=1, stroke=0)
            frac = max(0, min(1, (value-self.lo)/(self.hi-self.lo)))
            c.setFillColor(color)
            c.roundRect(label_w, y, chart_w*frac, 4.5*mm, 2.25*mm, fill=1, stroke=0)
            c.setFillColor(INK)
            c.setFont("BodyBold", 7.5)
            c.drawRightString(self.width, y + 1.2*mm, f"{value:.4f}")


def _fonts():
    candidates = [
        ("C:/Windows/Fonts/aptos.ttf", "C:/Windows/Fonts/aptosbd.ttf"),
        ("C:/Windows/Fonts/calibri.ttf", "C:/Windows/Fonts/calibrib.ttf"),
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ]
    for regular, bold in candidates:
        if Path(regular).exists() and Path(bold).exists():
            pdfmetrics.registerFont(TTFont("Body", regular))
            pdfmetrics.registerFont(TTFont("BodyBold", bold))
            return
    pdfmetrics.registerFont(TTFont("Body", "C:/Windows/Fonts/arial.ttf"))
    pdfmetrics.registerFont(TTFont("BodyBold", "C:/Windows/Fonts/arialbd.ttf"))


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("Title", parent=base["Title"], fontName="BodyBold",
            fontSize=28, leading=31, textColor=NAVY, alignment=TA_LEFT, spaceAfter=7*mm),
        "subtitle": ParagraphStyle("Subtitle", fontName="Body", fontSize=11.5,
            leading=16, textColor=MUTED, spaceAfter=7*mm),
        "h1": ParagraphStyle("H1", fontName="BodyBold", fontSize=19, leading=23,
            textColor=NAVY, spaceAfter=5*mm, spaceBefore=1*mm),
        "h2": ParagraphStyle("H2", fontName="BodyBold", fontSize=11.5, leading=14,
            textColor=TEAL, spaceBefore=4*mm, spaceAfter=2*mm),
        "body": ParagraphStyle("Body", fontName="Body", fontSize=8.8, leading=12.3,
            textColor=INK, spaceAfter=2.3*mm),
        "small": ParagraphStyle("Small", fontName="Body", fontSize=7.3, leading=9.5,
            textColor=MUTED, spaceAfter=1.6*mm),
        "callout": ParagraphStyle("Callout", fontName="BodyBold", fontSize=10.5,
            leading=14, textColor=NAVY, alignment=TA_CENTER),
        "metric": ParagraphStyle("Metric", fontName="BodyBold", fontSize=18,
            leading=21, textColor=TEAL, alignment=TA_CENTER),
        "metric_label": ParagraphStyle("MetricLabel", fontName="Body", fontSize=7.2,
            leading=9, textColor=MUTED, alignment=TA_CENTER),
        "ref": ParagraphStyle("Ref", fontName="Body", fontSize=7.2, leading=9.2,
            textColor=INK, leftIndent=5*mm, firstLineIndent=-5*mm, spaceAfter=1.6*mm),
    }


def P(text, s, style="body"):
    return Paragraph(text, s[style])


def metric_cards(items, s):
    cells = []
    for value, label in items:
        cells.append([P(value, s, "metric"), P(label, s, "metric_label")])
    table = Table([[Table([[c[0]], [c[1]]], colWidths=[47*mm]) for c in cells]],
                  colWidths=[51*mm]*len(cells))
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), PALE_TEAL),
        ("BOX", (0,0), (-1,-1), 0.6, colors.HexColor("#B8DBD7")),
        ("INNERGRID", (0,0), (-1,-1), 3, WHITE),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    return table


def data_table(rows, widths, s, header=True):
    cooked = [[P(str(x), s, "small") for x in row] for row in rows]
    t = Table(cooked, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("GRID", (0,0), (-1,-1), 0.35, LINE),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("ROWBACKGROUNDS", (0,1 if header else 0), (-1,-1), [WHITE, PAPER]),
    ]
    if header:
        commands += [("BACKGROUND", (0,0), (-1,0), NAVY),
                     ("TEXTCOLOR", (0,0), (-1,0), WHITE)]
    t.setStyle(TableStyle(commands))
    return t


def _header_footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setStrokeColor(LINE)
    canvas.line(20*mm, h-15*mm, w-20*mm, h-15*mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Body", 7)
    canvas.drawString(20*mm, h-11.5*mm, "LATENT PROBING FOR TOXICITY")
    canvas.drawRightString(w-20*mm, 11*mm, f"PREPRINT  |  {doc.page}")
    canvas.restoreState()


def build(output: Path, result_path: Path | None = None):
    _fonts()
    s = _styles()
    result = {}
    if result_path and result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    final_score = float(result.get("final_development_score", 0.8976470588))
    final_correct = int(round(1700*final_score))
    final_label = result.get("final_label", "gemma_pooled_std_lda06")
    candidate_score = result.get("router_score")
    router_disposition = result.get("router_disposition", "awaiting live evaluation")
    testing_score = result.get("testing_score")
    testing_correct = result.get("testing_correct")
    testing_rows = result.get("testing_rows", 1360)
    testing_submission_id = result.get("testing_submission_id")
    testing_rank = result.get("testing_rank_as_checked")

    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=20*mm,
        rightMargin=20*mm, topMargin=21*mm, bottomMargin=17*mm,
        title="Corpus Shift, Frozen Features: A Competition Report on Latent Toxicity Probing",
        author="Blessing Mambwe")
    story = []

    story += [Spacer(1, 14*mm), P("Corpus Shift,<br/>Frozen Features", s, "title"),
              P("A competition report on latent toxicity probing", s, "subtitle")]
    story.append(Table([[P("BLESSING MAMBWE", s, "h2"),
                         P("PREPRINT  |  10 SEPTEMBER 2026", s, "small")]],
                       colWidths=[95*mm, 70*mm], style=[("VALIGN",(0,0),(-1,-1),"BOTTOM")]))
    story += [Spacer(1, 9*mm), metric_cards([
        (f"{final_score:.4f}", "FINAL DEVELOPMENT ACCURACY"),
        (f"{final_correct}/1700", "CORRECT DEVELOPMENT ROWS"),
        ("+0.1247", "ABSOLUTE GAIN FROM v007"),
    ], s), Spacer(1, 9*mm)]
    story += [P("Abstract", s, "h1"),
              P("We study a constrained toxicity challenge in which participants receive fixed 2,304-dimensional hidden-state embeddings and must submit a pre-trained classifier without fitting on evaluation data. The strongest reliable system pools 12 cleaned public sources, standardizes each dimension using training statistics, fits shrinkage linear discriminant analysis, and applies a fixed 12/17 positive-rate operating point. It reaches 0.897647 development accuracy (1,526 of 1,700) and 0.923529 Testing accuracy (1,256 of 1,360), up from a 0.7729 development baseline. A fixed batch router improves macro leave-one-family-out accuracy from 0.795946 to 0.799313 and source-stratified accuracy from 0.829456 to 0.870673, but ties the pooled model live and is not used for Testing. External teacher filtering and ParaDetox augmentation fail to improve transfer. The main finding is methodological: under severe corpus shift, corpus construction and label semantics dominate small classifier variations.", s),
              Spacer(1, 3*mm),
              data_table([["Scope", "Evidence boundary"],
                          ["Observed", "CodaBench accuracy, offline accuracy/AUROC, artifact hashes, runtime checks"],
                          ["Inferred", "Development prior 1200/1700; likely but unconfirmed final prior"],
                          ["Not claimed", "Organizer corpus identity; leaderboard AUROC; causal explanation of the leader's score"]],
                         [34*mm, 131*mm], s), PageBreak()]

    story += [P("1. Challenge contract and evidence", s, "h1"),
              P("The TRI AI Cohort 10 project asks teams to build a binary toxicity detector from Gemma 2 2B hidden states. The official project page supplies no training corpus and explicitly makes data sourcing part of the task [1]. CodaBench evaluates a ZIP containing <font name='BodyBold'>classifier.py</font> and <font name='BodyBold'>trained_probe.joblib</font>; all learned coefficients must exist before upload.", s),
              P("The development batch contains 1,700 examples. Repeated threshold submissions with byte-identical weights revealed an exact 1,200-positive operating count, motivating a 12/17 quota. That value is an inference from development aggregates, not a published final-test fact.", s),
              P("What the leaderboard tells us", s, "h2"),
              BarChart([("v007 starting point", .7729, MUTED),
                        ("raw LDA ablation", .8765, ORANGE),
                        ("rank ensemble", .8918, ORANGE),
                        ("pooled std-LDA", .897647, TEAL),
                        ("current leader", .941176, NAVY)]),
              Spacer(1, 4*mm),
              P("Accuracy alone does not determine AUROC. Any conversion from leaderboard accuracy to an 'implied AUROC' requires an unverified score-distribution model. We therefore report AUROC only for offline examples with real-valued scores.", s),
              P("Submission safety", s, "h2"),
              data_table([["Invariant", "Implementation"],
                          ["No evaluation-time fitting", "Submission source is rejected if it contains .fit("],
                          ["Portable inference", "NumPy arithmetic; no serialized sklearn estimator"],
                          ["Force_Last protection", "Restore the proven ZIP immediately after any worse experiment"],
                          ["Determinism", "Smoke tests at 1,700 and 1,360 rows plus batch permutation checks"]],
                         [46*mm,119*mm], s), PageBreak()]

    story += [P("2. Corpus engineering", s, "h1"),
              P("Thirteen public toxicity sources were streamed, normalized, and globally de-duplicated before feature extraction. One single-class source was removed and correlated datasets were grouped into nine evaluation families. Each retained source was class-balanced independently so source identity could not act as a proxy for label.", s),
              metric_cards([("149,528", "RAW EXTRACTED ROWS"), ("108,468", "CLEAN TRAINING ROWS"),
                            ("12", "RETAINED SOURCES")], s), Spacer(1, 5*mm),
              data_table([
                  ["Family", "Retained examples", "Role / label semantics"],
                  ["Civil", "26,218", "Civil Comments plus unintended-bias sample; strict neutral cutoff"],
                  ["Twitter", "27,572", "Davidson and TweetEval hate/offensive tasks"],
                  ["Wiki", "11,314", "Jigsaw Wikipedia talk-page toxicity"],
                  ["Berkeley", "16,000", "Aggregated human hate-speech ratings"],
                  ["RealToxicityPrompts", "14,000", "Prompt halves labeled by toxicity threshold"],
                  ["Aegis", "6,152", "Content-safety majority labels"],
                  ["OffensiveLang", "3,496", "Implicit offensive statements"],
                  ["HateCheck", "2,326", "Functional, partly templated hate tests"],
                  ["ToxicChat", "1,390", "Conversation safety labels"],
              ], [39*mm, 29*mm, 97*mm], s),
              P("Leakage controls", s, "h2"),
              P("Normalized text is de-duplicated globally and family groups remain intact during validation. This prevents near-identical Civil, Twitter, or Wikipedia samples from appearing in both training and held-out folds. Three seeded class-prior resamples are reported only as sensitivity checks; because they overlap, they are not independent replications.", s),
              P("Exact-corpus search", s, "h2"),
              P("The organizer's public curriculum, repositories, Hugging Face catalog, and candidate fingerprints were searched. No exact corpus was identified. This negative result does not prove that the corpus is unavailable; it only bounds what was found in public evidence as of 9 September 2026.", s), PageBreak()]

    story += [P("3. Frozen representation and linear head", s, "h1"),
              P("Text is tokenized to 64 tokens and passed through the frozen Gemma 2 2B transformer [2,3]. The representation is the attention-mask-weighted mean of layer 14 hidden states, yielding one 2,304-dimensional vector per example. Masking avoids padding contamination and mean pooling avoids the brittle last-token convention.", s),
              P("Standardized shrinkage LDA", s, "h2"),
              P("For feature j, z_j = (x_j - mu_j) / max(sigma_j, epsilon), where mu and sigma are fitted on training data only. With class means m0 and m1 and a pooled covariance estimate S, the stored linear score is s(x) = w^T z + b, with w = S_lambda^{-1}(m1 - m0). The shrinkage setting lambda = 0.6 was selected by grouped transfer validation.", s),
              P("At inference, the proven model ranks all scores and labels the top round(12n/17) examples positive. This rank rule is insensitive to score calibration but assumes stable class balance. The ZIP contains one head, its train mean and scale, and no mutable state.", s),
              P("Why a linear probe", s, "h2"),
              P("Linear probes measure information accessible through a simple decision boundary [4]. They are also the competition-compatible choice: small, deterministic, fast, and auditable. Probe accuracy is not by itself a causal explanation of a model representation [5], so our conclusions concern predictive transfer, not mechanistic necessity.", s),
              data_table([["Component", "Frozen at submission?", "Uses labels?", "Batch-dependent?"],
                          ["Gemma embedding", "Yes", "No", "No"],
                          ["Standardization", "Yes - training moments", "No", "No"],
                          ["LDA coefficients", "Yes", "Yes - public corpus", "No"],
                          ["Quota decision", "Yes - 12/17", "No", "Yes - rank only"],
                          ["Optional router", "Yes - heads and moments", "No", "Yes - head selection"]],
                         [48*mm, 39*mm, 33*mm, 45*mm], s),
              P("The optional router reads only unlabeled batch moments to choose among stored heads. Conditional on that choice, prediction remains a fixed linear score plus quota. This is a rule interpretation, not organizer endorsement.", s), PageBreak()]

    story += [P("4. Model selection under corpus shift", s, "h1"),
              P("Grouped validation is substantially harder than the hidden development batch: the winning recipe averages 0.7934 leave-one-family-out accuracy but scores 0.897647 live. We use the former to reject fragile ideas, not as a calibrated forecast of the latter.", s),
              data_table([["Recipe", "Offline accuracy", "Offline AUROC", "Development accuracy"],
                          ["std | LDA | 0.6", "0.7934", "0.8462", "0.897647"],
                          ["rank ensemble", "0.7934", "-", "0.8918"],
                          ["raw | LDA | 0.6", "0.7903", "0.8563", "0.8765"],
                          ["ABTT family", "-", "0.6859 mean", "not resubmitted"],
                          ["nonlinear MLP", "-", "-0.022 vs linear", "not submitted"],
                          ["CORAL", "-", "-0.032 vs baseline", "not submitted"]],
                         [48*mm, 38*mm, 39*mm, 40*mm], s),
              P("Empirical trajectory", s, "h2"),
              BarChart([("all-positive baseline", .705882, MUTED),
                        ("initial threshold", .704706, ORANGE),
                        ("quota, same weights", .767059, ORANGE),
                        ("v007", .7729, ORANGE),
                        ("pooled std-LDA", .897647, TEAL)], lo=.68, hi=.91),
              Spacer(1, 4*mm),
              P("The largest gain came from changing the training distribution, not from elaborating the classifier. Rank ensembling lost 10 live rows and raw features lost 36. Those live ablations are more decision-relevant than tiny unpaired cross-validation differences.", s),
              P("Statistical reading", s, "h2"),
              P("For 1,526/1,700 correct, the ordinary binomial standard error is about 0.00735 and an approximate Wilson 95% interval is 0.8823 to 0.9112. This interval describes sampling uncertainty under an IID model, not leaderboard-selection bias. Comparing two submissions on the same hidden rows requires their paired disagreement table (for example, McNemar's test), which CodaBench does not expose.", s), PageBreak()]

    story += [P("5. Adaptive corpus routing", s, "h1"),
              P("Corpus shift suggested a fixed mixture-of-experts variant. For each source head, training-time class means and diagonal variances are stored. At inference, these moments are mixed at the configured prior and compared with the unlabeled batch mean and standard deviation using a pooled-scale-normalized diagonal distance. The closest source head is selected, with the exact pooled leaderboard model included as a fallback.", s),
              data_table([["Full-data LOSO method", "Accuracy", "AUROC", "Delta accuracy"],
                          ["Pooled baseline", "0.795946", "0.832204", "-"],
                          ["Router + pooled fallback", "0.799313", "0.838130", "+0.003367"],
                          ["Source-only router", "0.798921", "0.838564", "+0.002975"],
                          ["Drop HateCheck + Aegis", "0.793399", "0.829022", "-0.002547"]],
                         [61*mm, 34*mm, 34*mm, 36*mm], s),
              P("A source-stratified 80/20 diagnostic asks a different question: can batch moments recognize an in-family sample? The router selected the correct family for all nine families and raised macro accuracy from 0.829456 to 0.870673. HateCheck produced the largest change, 0.8121 to 0.9879, reflecting its deliberately functional distribution.", s),
              P("Interpretation", s, "h2"),
              data_table([["Evidence for", "Evidence against / unresolved"],
                          ["Batch moments contain strong family identity", "LOSO gain is only 0.34 percentage points"],
                          ["Stored source heads can exploit recognized families", "The hidden family is unknown"],
                          ["Pooled fallback limits gross mismatch", "Fallback does not guarantee incumbent predictions"],
                          ["No labels or fitted parameters at inference", "Rule permissibility has not been organizer-confirmed"]],
                         [82.5*mm, 82.5*mm], s),
              P("Live disposition", s, "h2"),
              P((f"The router scored <font name='BodyBold'>{float(candidate_score):.6f}</font> on development and was {router_disposition}."
                 if candidate_score is not None else
                 "At manuscript generation the router had no accepted live score. The proven pooled artifact therefore remained the final model."), s), PageBreak()]

    story += [P("6. External supervision did not transfer", s, "h1"),
              P("We tested two corpus interventions. First, unitary/toxic-bert at pinned revision 4d6c22e was used only as a training-data filter: retain a positive original label when teacher toxicity is at least 0.7 and a negative label when it is at most 0.3. Second, ParaDetox contributed human toxic-to-neutral rewrite pairs [6,7]. Evaluation labels were never filtered.", s),
              data_table([["Training method", "Accuracy", "AUROC", "Matched baseline"],
                          ["Original pooled", "0.795946", "0.832204", "0.795946"],
                          ["Teacher agreement", "0.785709", "0.815232", "0.795946"],
                          ["ParaDetox only", "0.702109", "0.696101", "0.781455"],
                          ["ParaDetox augmented", "0.781849", "0.811965", "0.781455"]],
                         [55*mm, 35*mm, 35*mm, 40*mm], s),
              P("ParaDetox hygiene", s, "h2"),
              metric_cards([("19,744", "RAW TOXIC-NEUTRAL PAIRS"), ("10,156", "OVERLAPS EXCLUDED"),
                            ("3,000", "BALANCED NEW ROWS")], s), Spacer(1, 4*mm),
              P("Civil and Wikipedia families were excluded from claims of independent ParaDetox transfer because the dataset shares Jigsaw ancestry. The retained sample fingerprint is bce9331fed368fb3. Neither teacher filtering nor ParaDetox met the submission gate: teacher filtering was clearly worse, while augmentation was flat in accuracy and lower in AUROC.", s),
              P("Lesson", s, "h2"),
              P("A high-capacity toxicity teacher is not automatically a better label oracle for a representation trained and evaluated under different semantics. Filtering removes hard boundary cases, shifts source mixtures, and can compress precisely the distinctions needed for transfer. Contrastive detoxification pairs likewise emphasize rewriting rather than the challenge's binary labeling boundary.", s), PageBreak()]

    story += [P("7. Final system and reproducibility", s, "h1"),
              metric_cards([(f"{final_score:.6f}", "FINAL DEVELOPMENT SCORE"),
                            ((f"{float(testing_score):.6f}" if testing_score is not None else "pending"), "TESTING SCORE"),
                            ("31/31", "LOCAL TESTS PASSED")], s), Spacer(1, 5*mm),
              P((f"The stable incumbent is identified by SHA-256 0bd11ac11c0a09a8b38d834e06c558f0e17236dae680c215a5e33af8c746a601. CodaBench Testing submission {testing_submission_id} finished at <font name='BodyBold'>{float(testing_score):.10f}</font> ({testing_correct}/{testing_rows}) and was rank {testing_rank} when checked on 10 September 2026. Testing remains open until 12 September, so placement is provisional."
                 if testing_score is not None else
                 "The stable incumbent is identified by SHA-256 0bd11ac11c0a09a8b38d834e06c558f0e17236dae680c215a5e33af8c746a601. The Testing score remains pending."), s),
              data_table([["Artifact / record", "Purpose"],
                          ["artifacts/kaggle/select/submissions/best_single.zip", "Proven pooled std-LDA model"],
                          ["artifacts/kaggle/corpus-transfer-full/transfer_results.json", "Router LOSO and source holdout evidence"],
                          ["artifacts/kaggle/teacher-select/teacher_select_results.json", "Teacher and ParaDetox rejection evidence"],
                          ["artifacts/kaggle/teacher-corpus/new_corpus_manifest.json", "New-data lineage and pinned teacher"],
                          ["RESEARCH_EXECUTION.md", "Chronological, corrected execution record"],
                          ["tests/test_routing.py", "Router determinism and permutation checks"]],
                         [75*mm, 90*mm], s),
              P("Reproduction outline", s, "h2"),
              P("1. Stream and normalize the declared public corpora. 2. Globally de-duplicate and rebalance each source. 3. Extract frozen Gemma layer-14 masked means. 4. Run grouped family validation. 5. Fit the selected std-LDA head on all cleaned rows. 6. Package only fixed arrays and inference code. 7. Execute isolated smoke tests at the expected development and final row counts. 8. Record the ZIP hash before upload.", s),
              P("Limitations", s, "h2"),
              P("The development score has been adaptively selected across several uploads and is therefore optimistic as a generalization estimate. Dataset licenses and original annotation policies differ. The inferred quota can fail under prior shift. Source-family validation is structurally mismatched to the hidden distribution. The corpus search was incomplete by construction, and competitor methods are unknown. Finally, a predictive probe demonstrates accessible signal, not a causal role for the probed representation.", s), PageBreak()]

    story += [P("References", s, "h1"),
              P("[1] TRI AI / AI Saturdays Lagos. Cohort 10 Projects: Latent Probing for Toxicity. https://aisaturdayslagos.github.io/cohort_structure/cohort10/projects.html", s, "ref"),
              P("[2] Team Gemma. Gemma 2: Improving Open Language Models at a Practical Size. arXiv:2408.00118, 2024. https://arxiv.org/abs/2408.00118", s, "ref"),
              P("[3] Google. Gemma 2 2B model card. https://huggingface.co/google/gemma-2-2b", s, "ref"),
              P("[4] Guillaume Alain and Yoshua Bengio. Understanding Intermediate Layers Using Linear Classifier Probes. ICLR Workshop, 2017. https://arxiv.org/abs/1610.01644", s, "ref"),
              P("[5] John Hewitt and Percy Liang. Designing and Interpreting Probes with Control Tasks. EMNLP-IJCNLP, 2019. https://aclanthology.org/D19-1275/", s, "ref"),
              P("[6] Mikhail Dementieva et al. ParaDetox: Detoxification with Parallel Data. ACL, 2022. https://aclanthology.org/2022.acl-long.469/", s, "ref"),
              P("[7] s-nlp. ParaDetox dataset card. https://huggingface.co/datasets/s-nlp/paradetox", s, "ref"),
              P("[8] Laura Hanu and Unitary team. Detoxify / toxic-bert model card and implementation notes. https://huggingface.co/unitary/toxic-bert", s, "ref"),
              P("[9] Daniel Borkan et al. Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification. WWW, 2019. https://arxiv.org/abs/1903.04561", s, "ref"),
              P("[10] Thomas Davidson et al. Automated Hate Speech Detection and the Problem of Offensive Language. ICWSM, 2017. https://ojs.aaai.org/index.php/ICWSM/article/view/14955", s, "ref"),
              P("[11] Paul Rottger et al. HateCheck: Functional Tests for Hate Speech Detection Models. ACL-IJCNLP, 2021. https://aclanthology.org/2021.acl-long.4/", s, "ref"),
              P("[12] Samuel Gehman et al. RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models. Findings of EMNLP, 2020. https://aclanthology.org/2020.findings-emnlp.301/", s, "ref"),
              P("[13] CodaBench competition 17670. Rules, phases, and leaderboard. https://www.codabench.org/competitions/17670/", s, "ref"),
              Spacer(1, 6*mm),
              Table([[P("Research note", s, "callout")],
                     [P("All numerical claims in this manuscript are tied to saved artifacts or the public leaderboard. Failed experiments are retained because negative transfer is part of the result.", s)]],
                    colWidths=[165*mm], style=[("BACKGROUND",(0,0),(-1,0),PALE_ORANGE),
                    ("BACKGROUND",(0,1),(-1,-1),PAPER),("BOX",(0,0),(-1,-1),0.6,ORANGE),
                    ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
                    ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7)])]

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return output


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("output/pdf/latent_probe_toxicity_preprint.pdf"))
    ap.add_argument("--result", type=Path, default=Path("artifacts/final_submission_result.json"))
    args = ap.parse_args()
    print(build(args.output, args.result))
