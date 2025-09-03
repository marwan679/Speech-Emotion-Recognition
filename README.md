# English Speech Emotion Recognition (SER) — RAVDESS + Keras/LSTM

This project trains and serves an **English speech emotion recognition** (SER) model on the **RAVDESS** dataset.  
It walks through **data loading, preprocessing (MFCCs), optional augmentation, model building (TimeDistributed Conv1D → LSTM), training (with callbacks), evaluation (confusion matrix), and inference from files or microphone** — plus an optional **gender classifier** using a pre‑trained SVM.

> Notebook: `english speech model.ipynb`

---

## ✨ What you’ll build
- A supervised 8‑class SER model: **neutral, calm, happy, sad, angry, fearful, disgust, surprised**.
- Input features: **20 MFCCs** over ~2.4‑second clips (≈104 frames) extracted with `librosa`.
- Architecture: **TimeDistributed(Conv1D + BN + Flatten) → LSTM(64) → Dense** → Softmax(8).
- Training aids: **ReduceLROnPlateau**, **EarlyStopping**, **ModelCheckpoint**; optional **TensorBoard**.
- Evaluation: accuracy + **confusion matrix heatmap**.
- Inference utilities: predict from `.wav`/`.mp3` **files** or from **microphone** (`sounddevice`).  
- Optional: **gender detection** using a pre‑trained **SVM** on MFCC features (`svm_model.pkl`).

---

## 📦 Dataset
- **RAVDESS**: “Ryerson Audio‑Visual Database of Emotional Speech and Song”.  
  Get it from Zenodo or Kaggle, then put the **speech audio** subset in the project root (or keep the provided ZIP).  
  - Zenodo (full official release)  
    https://zenodo.org/record/1188976
  - Kaggle (speech audio-only subset)  
    https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

**File naming convention (how labels are parsed):** recordings are named like `03-01-06-01-02-01-12.wav` where the 3rd code = **emotion** (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised). The notebook reads this field to assign labels.

---

## 🛠️ Environment & dependencies

Create a fresh Python env (3.9–3.11 is fine), then install:

```bash
pip install tensorflow keras librosa soundfile matplotlib seaborn scikit-learn tqdm pydub sounddevice scipy pandas
# Optional (for TensorBoard UI)
pip install tensorboard
```

> **Note on audio backends**
> - `librosa` needs the `soundfile` backend (package name: `pysoundfile`). Installing `soundfile` via `pip` covers it.
> - `pydub` requires **ffmpeg** on your system for MP3 support:  
>   - Ubuntu/Debian: `sudo apt-get install ffmpeg`  
>   - Windows: download ffmpeg and add its `bin/` to PATH  
> - `sounddevice` depends on **PortAudio** (pre‑built wheels included for most OSes). On Linux you may also need `libportaudio2` from your package manager.

---

## 📁 Project layout (suggested)
```
.
├── english speech model.ipynb
├── speech-emotion-recognition-ravdess-data.zip     # optional: auto-unzipped by the notebook
├── speech-emotion-recognition-ravdess-data/        # or keep the extracted WAVs here
│   ├── Actor_01/ ...
│   └── Actor_24/ ...
├── best.keras                                      # saved best SER model (created by training)
├── model.keras                                     # (if you choose to save under this name)
├── svm_model.pkl                                   # optional: pre-trained gender SVM
└── logs/fit/                                       # TensorBoard logs (optional)
```

---

## 🔄 End‑to‑end pipeline (what the notebook does)

1. **Unzip dataset (if needed)**
   - Helper `unzip_file(zip_filepath, extract_to)` expands `speech-emotion-recognition-ravdess-data.zip` to a folder.

2. **Enumerate audio + derive labels**
   - `load_data(path)` walks `speech-emotion-recognition-ravdess-data/Actor_*` folders, collects file paths, and extracts the **emotion id** from the filename’s 3rd field.
   - Produces two aligned lists: `pathes` and `emotions` (1..8).

3. **Signal loading & (optional) augmentation**
   - `librosa.load(path, duration=2.4, offset=0.6)` trims to a centered 2.4‑s slice for consistency.
   - Sample augmentation utilities are provided: `add_noise`, `pitch`, `stretch` (the notebook mainly demos **pitch shift**). You can insert them before feature extraction to enlarge data.

4. **Feature extraction (MFCCs)**
   - `get_features(path)` → reads audio → computes **MFCCs (n_mfcc=20)** and organizes them as time‑steps × coefficients.
   - A loop builds arrays **X** (stack of MFCC sequences) and **y** (one‑hot vectors for the 8 emotions).

5. **Split & shape**
   - `train_test_split(X, y, test_size=0.2, shuffle=True, random_state=11)`.
   - Add a channels dimension and **swap axes** to match Keras’ expected time‑major shape: final shape ≈ `(N, 104, 20, 1)`.

6. **Model**
   ```python
   def createmodel(input_shape):
       model = keras.Sequential(name="LSTM")
       model.add(TimeDistributed(Conv1D(32, 3, padding="same", activation="relu"),
                                 input_shape=input_shape))
       model.add(TimeDistributed(BatchNormalization()))
       model.add(TimeDistributed(Flatten()))
       model.add(LSTM(64))
       model.add(Dropout(0.2))
       model.add(Dense(64, activation="relu"))
       model.add(Dropout(0.2))
       model.add(Dense(64, activation="relu"))
       model.add(Dense(8, activation="softmax"))
       return model
   ```

7. **Compile & train**
   - Optimizer: `Adam(learning_rate=0.01)`  
   - Loss: `categorical_crossentropy`, Metric: `accuracy`
   - Callbacks:  
     - `ReduceLROnPlateau(monitor="val_loss", factor=0.6, patience=5, min_lr=1e-8)`  
     - `EarlyStopping(monitor="val_loss", patience=7)`  
     - `ModelCheckpoint(filepath="best.keras")`
   - Fit: `model.fit(train_x, y_train, batch_size=140, epochs=60, validation_data=(test_x, y_test), callbacks=[...])`
   - Optional: enable TensorBoard logging (`logs/fit`) and run `%tensorboard --logdir "logs/fit"` in the notebook.

8. **Evaluate & visualize**
   - `model.evaluate(test_x, y_test)` prints accuracy.
   - Predictions → `confusion_matrix(y_true, y_pred)` and a **Seaborn heatmap** to inspect per‑class performance.

9. **Inference utilities**
   - **From file**: `emotion_detector(path)` → extracts features → loads the model → returns the **emotion string**.
   - **From mic**: `voice_recorder(seconds, "record.wav")` records audio, then call `emotion_detector("record.wav")`.
   - **Optional gender**: `detect_gender(path)` loads an SVM (`svm_model.pkl`), extracts **40‑MFCC mean features**, scales them, and predicts **"male"/"female"`.


---

## ▶️ Quickstart

### 1) Prepare data
- Put the ZIP in the repo root **or** place the extracted WAVs under `speech-emotion-recognition-ravdess-data/`.

### 2) Train
Open the notebook and execute all cells. A `best.keras` file will be created (the top‑performing checkpoint).

### 3) Evaluate
View printed accuracy and the confusion matrix heatmap.

### 4) Inference
```python
# Load best model and predict on a file
emotion = emotion_detector("path/to/sample.wav")
print("Emotion:", emotion)

# Record 10 seconds and predict
voice_recorder(10, "record.wav")
print("Emotion:", emotion_detector("record.wav"))
print("Gender :", detect_gender("record.wav"))   # Optional, requires svm_model.pkl
```

---

## ✅ Tips & common gotchas

- **Model name mismatch**: training saves `best.keras`, but some inference snippets load `model.keras`. Make sure you load the same filename you saved (or change `filepath` in `ModelCheckpoint`).
- **MP3 support**: for `.mp3` inputs, `pydub` converts to WAV, but **ffmpeg must be installed** and on PATH.
- **Sound device permissions**: on macOS/Linux, allow microphone access if recording fails. On Linux you may need `sudo apt-get install libportaudio2`.
- **Librosa backend**: if you get `sndfile` errors, ensure `pip install soundfile` succeeded.
- **Reproducibility**: set `numpy`/`tensorflow` seeds if you need deterministic runs.
- **GPU**: TensorFlow will automatically use CUDA/cuDNN if properly installed; otherwise it runs on CPU.

---

## 📈 Extending the notebook

- Swap MFCCs for **Mel‑spectrograms** or add **delta MFCCs**.
- Try 2‑D Conv layers over (time × features) without `Flatten`, then LSTM/GRU.
- Swap `LSTM(64)` for **BiLSTM**, **GRU**, or **Transformer encoder** blocks.
- Add **SpecAugment**‑style masking for robustness.
- Calibrate the model with a **validation split** and consider **class weighting**.

---

## 🔗 References & further reading

- RAVDESS Dataset (official): https://zenodo.org/record/1188976  
- RAVDESS paper (PLOS ONE, 2018): https://doi.org/10.1371/journal.pone.0196391  
- Kaggle RAVDESS speech subset: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio  
- Librosa MFCCs: https://librosa.org/doc/main/generated/librosa.feature.mfcc.html  
- Keras TimeDistributed: https://keras.io/api/layers/recurrent_layers/time_distributed/  
- Keras LSTM: https://keras.io/api/layers/recurrent_layers/lstm/  
- Keras Adam: https://keras.io/api/optimizers/adam/  
- Keras Callbacks (ReduceLROnPlateau, EarlyStopping, ModelCheckpoint):  
  https://keras.io/api/callbacks/reduce_lr_on_plateau/  
  https://keras.io/api/callbacks/early_stopping/  
  https://keras.io/api/callbacks/model_checkpoint/  
- Scikit‑learn confusion_matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html  
- Seaborn heatmap: https://seaborn.pydata.org/generated/seaborn.heatmap.html  
- sounddevice docs: https://python-sounddevice.readthedocs.io/  
- SciPy `wavfile.write`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html  
- tqdm: https://tqdm.github.io/  
- pydub (requires ffmpeg): https://github.com/jiaaro/pydub
```

# End of README.md
