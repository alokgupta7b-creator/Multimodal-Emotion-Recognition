# Phase 10 Warning Audit

## Scope and commands

The warning audit was performed without changing model weights, datasets,
preprocessing, or dependency versions.

Commands:

```text
python -W always -m pytest api -q -ra
python -W always -m pytest -q -ra
python -W always -m pytest speech_emotion/tests/test_feature_extraction.py::TestFeatureExtraction::test_short_audio_signal_padded speech_emotion/tests/test_metrics.py::TestMetrics::test_serialization_and_markdown speech_emotion/tests/test_training_pipeline.py::TestTrainingPipeline::test_mlp_early_stopping_handles_string_emotion_labels -q -ra
```

Results:

- Backend: `9 passed, 3 warnings`.
- Full project suite: `216 passed, 6 warnings`.
- Warning-focused tests: `3 passed, 3 warnings`.

## Findings

| # | Exact warning | Location | Classification | Impact |
|---|---|---|---|---|
| 1 | `PendingDeprecationWarning: Please use import python_multipart instead.` | `starlette/formparsers.py:12`, `import multipart` | Dependency warning | No effect on inference, uploads, cleanup, security, or test validity. |
| 2 | `DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute` | `<frozen importlib._bootstrap>:488` while loading the text/audio transformer stack | Dependency warning | Native-extension metadata warning only; model loading and inference succeeded. |
| 3 | `DeprecationWarning: builtin type SwigPyObject has no __module__ attribute` | `<frozen importlib._bootstrap>:488` while loading the text/audio transformer stack | Dependency warning | Native-extension metadata warning only; model loading and inference succeeded. |
| 4 | `UserWarning: n_fft=2048 is too large for input signal of length=1280` | `librosa/core/spectrum.py:266`, triggered by `speech_emotion/tests/test_feature_extraction.py:112-118` | Runtime warning | The short-input regression test completes successfully. This is an edge-case warning from librosa, not a failed extraction or API behavior. |
| 5 | `UserWarning: y_pred contains classes not in y_true` | `sklearn/metrics/_classification.py:2801`, triggered by `speech_emotion/tests/test_metrics.py:58-67` | Test-framework warning | The test intentionally checks predictions containing a class absent from the small true-label sample. Fixed seven-class metrics and serialization remain correct. |
| 6 | `ConvergenceWarning: Stochastic Optimizer: Maximum iterations (5) reached and the optimization hasn't converged yet.` | `sklearn/neural_network/_multilayer_perceptron.py:781`, triggered by `speech_emotion/tests/test_training_pipeline.py:124-139` | Test-framework warning | The test intentionally uses `max_iter=5` for a fast compatibility check. It does not exercise a production training configuration and does not affect inference. |

The first three entries are dependency/native-extension warnings. The last
three are emitted by intentional regression-test inputs. In addition, Python
prints `DeprecationWarning: builtin type swigvarlink has no __module__ attribute`
at `<sys>:0` during interpreter teardown after the pytest summary. It is the
same native dependency family, is not counted by pytest's six-warning summary,
and has no project source location or observed runtime impact.

## Investigation and decisions

### Starlette multipart import

Starlette 0.38.6 imports the compatibility module name `multipart` and emits
the pending deprecation warning. The application already declares
`python-multipart==0.0.12`, and changing FastAPI/Starlette versions solely to
remove this warning would be an unnecessary compatibility risk. The warning
does not indicate an upload failure. It remains documented.

### SWIG native warnings

The warnings are emitted at the importlib/native-extension boundary while
loading the existing transformer predictors. They do not identify a project
source line and are not produced by project code. Direct model construction
completed for text and audio, and the video predictor loaded without these
warnings. No safe application-level fix exists; suppressing them would hide
dependency diagnostics, so they remain documented.

### Short MFCC input

The test deliberately passes a 1,280-sample signal to the default 2,048-point
MFCC configuration. Librosa warns but returns finite features, and the test
passes. Changing padding behavior during this audit could alter the existing
MFCC preprocessing contract and artifacts, so no preprocessing change was
made.

### Metrics edge case

The metric test deliberately uses a prediction label that is absent from its
small `y_true` sample while the implementation requests the canonical
seven-class label order. Scikit-learn reports this informative warning; the
result is still serialized with all seven classes and zero-valued absent
supports. No production defect was found.

### MLP convergence edge case

The compatibility test deliberately limits the MLP to five iterations. The
warning accurately reports that this tiny test run did not converge. It is not
a production model run, and increasing iterations would make the test less
focused without changing the behavior under test.

## Final validation

No warning suppression, dependency upgrade, dataset modification, model
training, or test-cache generation was performed. The existing real text,
audio, and video endpoint tests remained passing. The warnings are
non-blocking and documented as dependency or intentional test-input
limitations.
