# Self-recorded Mandarin tone experiment results
## Main model comparison
| Dataset / model | Accuracy | Macro-F1 |
|---|---:|---:|
| THCHS raw acoustic | 0.2250 | 0.1983 |
| THCHS raw wav2vec2 | 0.3125 | 0.3149 |
| THCHS raw fusion | 0.2625 | 0.2599 |
| Self raw acoustic | 0.3639 | 0.3198 |
| Self raw wav2vec2 | 0.3389 | 0.3211 |
| Self raw fusion | 0.3556 | 0.3373 |
| Self filtered acoustic | 0.3761 | 0.3275 |
| Self filtered wav2vec2 | 0.3644 | 0.3354 |
| Self filtered fusion | 0.3965 | 0.3722 |

## Filtered leave-one-speaker-out acoustic baseline
| Test speaker | Accuracy | Macro-F1 |
|---|---:|---:|
| speaker01 | 0.5156 | 0.5028 |
| speaker02 | 0.3276 | 0.3186 |
| speaker03 | 0.3761 | 0.3275 |

## Paper notes
- THCHS tone-slice results are retained as a public continuous-speech slicing baseline and limitation analysis.
- The self-recorded dataset provides a controlled isolated-character setting with speaker-independent evaluation.
- Quality filtering improves the best fusion model from macro-F1 0.3373 to 0.3722, but the task remains difficult under unseen-speaker testing.
- Tone 4 remains the weakest category after filtering, while many samples are confused with Tone 3.
