# Sign2Text Offline

Local, webcam-based recognition for a limited set of hand gestures and ASL handshapes. The interface supports Russian and English. This is an experimental MVP, **not** general sign-language translation.

## What works

- **Gestures:** 26 app-defined handshapes, including simultaneous signs from two hands.
- **ASL numbers:** 1–9 in a separate recognition mode.
- **ASL fingerspelling:** A–Z. Recognized letters can be joined into words, with Space, Repeat, and Delete controls. J and Z require hand motion.
- Live webcam recognition, short recorded clips, and uploaded videos (1–10 seconds, up to 32 MB).
- A current-result view, an in-memory history, and visual guides for supported signs.
- A Russian/English interface switch. Gesture and number labels are translated; ASL letters and assembled English spellings remain Latin text.

The app runs its inference locally. Its server binds to `127.0.0.1` and does not expose the camera or inference API to the network. Model files are downloaded only during setup and verified with SHA-256 checksums.

## Quick start

Requires Python 3.12 on macOS or Linux. An internet connection is needed for the initial dependency and model download; inference then runs locally.

```bash
python3 -m venv work/model-env
work/model-env/bin/python -m pip install -r ml/requirements-asl-letters.txt
work/model-env/bin/python ml/install_gestures.py
work/model-env/bin/python ml/install_asl_letters.py
python3 start.py
```

Open [http://127.0.0.1:4173/](http://127.0.0.1:4173/) on the same computer. Grant camera access when prompted. To install only gesture and number recognition, use `ml/requirements-gestures.txt` and `ml/install_gestures.py`; the alphabet mode requires the additional PyTorch dependency and checkpoint.

The local Mac desktop app shown in some development screenshots is a machine-specific launcher and is intentionally not included in Git. The source app starts with the command above.

## Accuracy and limitations

Recognition is restricted to the supported guide, not arbitrary gestures, ASL sentences, or other sign languages. The simple gestures have app-defined English meanings; they are not necessarily conventional ASL signs. The interface rejects uncertain or conflicting signs rather than inventing a translation. It does not translate arbitrary fingerspelled English words into Russian.

Webcam accuracy has **not** been established across people, camera angles, or lighting. The tests verify the recognition pipeline and selected labeled examples, not a universal accuracy score. Two-hand recognition and moving J/Z have especially limited real-user validation. Do not use this prototype for safety-critical interpretation.

## Tests

```bash
work/model-env/bin/python -m unittest discover -s tests -v
node --test tests/test_client.cjs tests/test_adapter.cjs tests/test_i18n.cjs
```

`ml/example_predictions.jsonl` contains synthetic evaluation examples. Their metrics are not model accuracy. The research protocol for a future pose-only versus RGB-plus-pose comparison is in [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md); those two baselines have not been trained.

## Project structure

```text
app/       Browser interface and localhost HTTP server
ml/        Local recognizers, pinned model installers, evaluation tools
tests/     Client, API, data, and recognition tests
docs/      Research protocol
work/      Ignored virtual environment, downloaded weights, temporary video
outputs/   Ignored local development previews and reports
```

Video is processed by the local server. During recorded-clip processing, a temporary file is placed in `work/incoming` and deleted when the job finishes; a crash can leave an orphaned file there. No uploaded video, model weight, virtual environment, or private recording is included in this repository.

## Models, data, and licensing

The app downloads the [official MediaPipe gesture model](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer/python) and an [ASL landmark classifier checkpoint](https://github.com/TharunCh-tch/asl-detection) during setup. Their original authors' licenses and terms apply to those files. The archived experimental continuous-ASL adapter in `ml/runtime.py` references a separate [Sign Language Bridge model](https://huggingface.co/mamounyosef/sign-language-bridge); that multi-GB model is not the active MVP and is not distributed here. How2Sign data is likewise not included; obtain and use it only under its own terms.

The original code and documentation in this repository are released under the [MIT License](LICENSE). That license does **not** relicense third-party model weights, datasets, reference images, or videos.
