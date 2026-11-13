# Audio-Driven Hate Speech Detection in Telugu

Detecting hateful and offensive content directly from **Telugu speech**, and comparing audio-only, text-only and fused (audio + text) models on a multi-modal dataset.

An exploratory study of hate speech detection in Telugu social-media data, with an analysis of how much each modality (speech vs. transcript) contributes across hate-speech categories.

---

## Highlights

- **Two tasks:** binary (Hate / Non-Hate) and 5-class (Gender, Religion, Political, Character Assassination, Non-Hate).
- **Audio branch:** handcrafted OpenSMILE features (SVM, Gradient Boosting) vs. pretrained speech models (Wav2Vec2 XLS-R, Wav2Vec2 sequence classifier, Audio Spectrogram Transformer).
- **Text branch:** TF-IDF + SVM vs. multilingual encoders (mBERT, XLM-RoBERTa, LaBSE) on manual Telugu transcripts.
- **Multimodal branch:** early fusion (feature concatenation) and late fusion (probability averaging) of OpenSMILE audio features and LaBSE text embeddings.
- **Best results:** early fusion reaches **89.83%** binary accuracy; late fusion reaches **79.66%** 5-class accuracy.

---

## Dataset

Multi-modal Telugu hate speech data collected from YouTube: about 1 hour of hate speech and 1 hour of non-hate speech, with a **manual transcript for every audio sample**. Hate classes follow the YouTube Hate Speech Policy.

| Category | Speakers | Utterances |
| --- | --- | --- |
| Religion | 32 | 82 |
| Political / National Affiliation | 22 | 68 |
| Gender | 56 | 111 |
| Character Assassination | 46 | 132 |
| Non-Hate | 72 | 208 |

Vocabulary: 3,903 unique words in hate speech, 3,772 in non-hate, 6,975 overall.

> The dataset is **not included** in this repository. Place it under `TELUGU DATASET/TE/` (folders `Hate/{Gender,Political,Religion,Character Assassination}`, `Non Hate`, and `Excel` for metadata) before running the notebooks.

---

## Pipeline

```
Dataset ──┬── Speech ──► resample 16 kHz ──► OpenSMILE / Librosa / Wav2Vec2 / AST ──┐
          ├── Text ────► clean + normalise ─► TF-IDF / mBERT / XLM-R / LaBSE ───────┼─► Train & evaluate
          └── Both ────► shared stratified split ─► early fusion / late fusion ─────┘   (binary + 5-class)
```

**Audio preprocessing**
- Metadata extraction, class/duration analysis and duration-outlier checks.
- Resampling of all audio (mostly 44.1 kHz, one 48 kHz file) to a common **16 kHz**, mono.
- Train/test split (80/20), then **augmentation on the training set only** (time shift, time stretch, additive noise) to balance the minority classes up to 166 samples per class.

**Audio features and models**
- **OpenSMILE (ComParE 2016 low-level descriptors):** ZCR, RMS energy, voicing and MFCC 1-12, summarised per file with 10 statistics each (max, max position, range, median, kurtosis, skewness, std, two linear-regression coefficients, quadratic error) into a **150-dimensional vector**; classified with SVM (RBF) and Gradient Boosting.
- **Wav2Vec2 XLS-R 300M:** frozen mean-pooled embeddings + LSTM / 1D-CNN heads.
- **Wav2Vec2 sequence classifier:** fine-tuned end to end.
- **Audio Spectrogram Transformer (AST):** `MIT/ast-finetuned-audioset-10-10-0.4593`, fine-tuned.

**Text models:** TF-IDF + linear SVM, fine-tuned mBERT, mBERT embeddings + XGBoost, fine-tuned XLM-RoBERTa, LaBSE.

**Multimodal:** OpenSMILE audio features + LaBSE text embeddings, on a shared stratified split (589 matched audio-text pairs, 118 test records). Early fusion concatenates features; late fusion averages per-modality probabilities.

---

## Results

Test set: 121 records for unimodal models, 118 for the multimodal comparison.
Binary F1 is reported as in the project notebooks (positive-class F1 for the OpenSMILE rows, macro-F1 for the rest); 5-class F1 is macro-F1.

### Audio modality

| Model | Binary Acc | Binary F1 | 5-class Acc | 5-class F1 |
| --- | --- | --- | --- | --- |
| OpenSMILE + SVM | 81.82% | 0.8690 | 63.64% | 0.5859 |
| OpenSMILE + Gradient Boosting | 81.82% | 0.8659 | 59.50% | 0.5234 |
| Wav2Vec2 XLS-R + LSTM | 67.77% | 0.4851 | 43.80% | 0.4079 |
| Wav2Vec2 XLS-R + 1D-CNN | 65.29% | 0.3950 | 42.15% | 0.3122 |
| Wav2Vec2 sequence classifier | 77.69% | 0.7622 | 45.45% | 0.3935 |
| **Audio Spectrogram Transformer** | **85.95%** | 0.8441 | **70.25%** | 0.6229 |

### Text modality

| Model | Binary Acc | Binary F1 | 5-class Acc | 5-class F1 |
| --- | --- | --- | --- | --- |
| TF-IDF + linear SVM | 85.12% | 0.8393 | 80.17% | 0.7758 |
| mBERT fine-tuned | 81.82% | 0.8125 | 65.29% | 0.5855 |
| mBERT embeddings + XGBoost | 80.17% | 0.7835 | 62.81% | 0.5472 |
| XLM-RoBERTa fine-tuned | 81.82% | 0.8114 | 70.25% | 0.5916 |
| **LaBSE** | **89.26%** | 0.8794 | **82.64%** | 0.7929 |

### Multimodal (shared split: 589 pairs, 118 test)

| Task | Best strategy | Accuracy | Macro-F1 | Gain over LaBSE alone |
| --- | --- | --- | --- | --- |
| Binary | Early fusion | 89.83% | 0.8865 | +2.54 pts |
| 5-class | Late fusion | 79.66% | 0.7798 | +4.24 pts |

### Key findings

- Binary detection is consistently easier than 5-class categorisation.
- LaBSE is a strong text baseline; AST is the strongest audio deep model.
- OpenSMILE features stay competitive with far simpler computation than the pretrained speech models.
- Aggregate accuracy hides weak classes (e.g. XLM-RoBERTa has zero recall on Religion, LaBSE reaches 0.44).
- Fusion helps, and the best strategy depends on the task: early fusion for binary, late fusion for 5-class.

---

## Repository structure

| Notebook | Purpose |
| --- | --- |
| `audio_modality_preprocessing.ipynb` | Dataset exploration, 16 kHz resampling, train/test split, augmentation |
| `feature_extraction_svm_GB.ipynb` | OpenSMILE features + SVM / Gradient Boosting |
| `audio_modality_model_training.ipynb` | Wav2Vec2 XLS-R (LSTM / 1D-CNN), Wav2Vec2 classifier, AST |
| `text_modality.ipynb` | Text models: TF-IDF, mBERT, XLM-RoBERTa, LaBSE |
| `multi_modality` | Multimodal early / late fusion and final comparison |

Rename the notebooks to match this table if you prefer descriptive names.

---

## Getting started

The notebooks were developed on **Google Colab** (GPU recommended for the Wav2Vec2 and AST experiments) with the dataset stored on Google Drive.

```bash
pip install librosa soundfile torchaudio torch transformers datasets \
            opensmile scikit-learn xgboost pandas numpy scipy \
            matplotlib seaborn joblib openpyxl
```

Run the notebooks in order (1 → 5). Update the dataset, Drive and metadata paths at the top of each notebook to match your setup.

**Tech stack:** Python, PyTorch, Hugging Face Transformers, OpenSMILE, Librosa, scikit-learn, XGBoost, Pandas, NumPy, Matplotlib, Seaborn.

---

## Limitations and future work

- Small dataset (about 2 hours of audio) that may not cover all forms of Telugu hate speech; class imbalance affects category-wise performance.
- Splits are not speaker-independent and results come from a single seed, so speaker overlap may inflate scores.
- Noise, pronunciation and speaker variation affect audio models; transcript errors affect text and multimodal models.
- Next steps: speaker-independent splits, multiple seeds or cross-validation with confidence intervals, ablations over audio/text/fusion methods, Indic-language speech models, and noisy, short and code-mixed Telugu speech.

---


## License and data note
The dataset belongs to its original authors and is not redistributed in this repository.
