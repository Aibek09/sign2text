# Experiment plan: ASL → English

Status: this is a proposed controlled research comparison, not two implemented UI
modes. The current app instead integrates one experimental general-topic ASL
checkpoint (Sign Language Bridge / Qwen3-VL-2B). Café-only evaluation below is an
optional scoped experiment and must not be presented as broad-domain validation.

## MVP question

Can a local model translate 3–8 second ASL clips about ordering in a café, and how much do RGB frames add beyond hands/face/body keypoints?

This scope is deliberately narrower than open-domain sentence translation. Target utterances include ordering, quantities, sizes, payment, common questions, thanks, and simple corrections.

## Comparable model tracks

Both tracks must use the same train/dev/test split, tokenizer, decoder budget, maximum clip duration, vocabulary, random seeds, and hardware.

### A — pose

Input per frame: left/right hand landmarks, face landmarks (or a compact face subset), and upper-body landmarks. Normalize coordinates by shoulder midpoint and shoulder width; retain confidence/missingness masks. Use a temporal encoder followed by an autoregressive or CTC text decoder.

### B — RGB + pose

Use the same pose stream plus a compact visual encoder over sampled RGB frames. Fuse the streams before the temporal text decoder. This track may capture handshape, occlusion, objects, mouthings, and scene context that keypoints discard.

Start with frozen visual features to keep local training and inference practical. Only unfreeze after the baseline is reproducible.

## Data protocol

1. Obtain How2Sign metadata and videos under its official terms.
2. Run `ml/prepare_manifest.py --topic cafe` to create a café-domain subset
   (the tool defaults to all topics without this option).
3. Split by signer/video source, not random clip, to reduce identity leakage.
4. Deduplicate near-identical English text before reporting results.
5. Keep an untouched robustness test set.

The keyword filter is a bootstrap, not ground truth. Manually review the manifest and label each clip as in-domain, ambiguous, or out-of-domain.

## Metrics

Primary:

- WER (word error rate; lower is better)
- CER (character error rate; lower is better)
- exact sentence match
- end-to-end latency p50/p95 on the target laptop

Robustness slices:

- frontal / 20–30° side angle
- normal / dim / backlit illumination
- plain / visually busy background
- signer seen / unseen during training

Report the robustness gap as `slice WER − clean WER`, not only the slice score. Also record failure categories: omitted negation, wrong entity/quantity, handshape confusion, temporal boundary error, hallucinated fluent text, and no prediction.

## Evaluation JSONL

Each line accepted by `ml/evaluate.py`:

```json
{"id":"clip-001","reference":"I would like coffee","prediction":"I like coffee","latency_ms":380,"condition":"clean","mode":"pose"}
```

Use `condition` values such as `clean`, `side_angle`, `dim`, and `backlit`. Run each model on the same clip identifiers.

## Milestones

1. **Data audit:** downloadable subset, reviewed manifest, signer-safe split, license notes.
2. **Oracle pipeline:** feed ground-truth keypoints to a tiny decoder and verify training/evaluation plumbing.
3. **Pose baseline:** real keypoint extraction, temporal model, text decoding, local export.
4. **RGB + pose:** matched fusion baseline and ablation report.
5. **Robustness:** collect controlled angle/light variants of consented test phrases.
6. **Product integration:** export ONNX, connect the browser adapter, measure true end-to-end latency.

## Release gate for calling it a translator

Do not remove the “model not loaded” disclosure until a model produces outputs from unseen video. Do not claim general ASL support from the café subset. Publish results with signer-safe splits, sample counts, confidence intervals where practical, and visible failure examples.
