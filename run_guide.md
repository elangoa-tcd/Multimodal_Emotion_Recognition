# How to Run the TMBL Model on the Remote GPU

**Remote machine:** `elangoa@134.226.38.166`
**GPU:** NVIDIA RTX A6000 (49 GB VRAM)
**OS:** Ubuntu/Debian Linux

---

## Section 1 — Connect to the Remote Machine

### Option A: VS Code (recommended)
1. Open VS Code
2. Click the green **><** button at the bottom-left corner
3. Select **Connect to Host**
4. Type `elangoa@134.226.38.166` and press Enter
5. Enter your password when prompted
6. Open a terminal: `Ctrl + `` `

### Option B: PowerShell on your laptop
```powershell
ssh elangoa@134.226.38.166
```

---

## Section 2 — Set Up the Environment (one-time only)

> **Note:** You do not have sudo access on this machine. Ask your IT administrator to run:
> `sudo apt install python3.13-venv git python3-pip build-essential wget unzip -y`
> This is the only step that requires admin help — everything after uses your own home directory.

### 2.1 — Activate the virtual environment
```bash
source ~/tmbl_env/bin/activate
```

> **Every time you reconnect via SSH**, run this first before doing anything else.
> To make it automatic, add it to `~/.bashrc`:
> ```bash
> echo "source ~/tmbl_env/bin/activate" >> ~/.bashrc
> source ~/.bashrc
> ```

### 2.2 — Install PyTorch with CUDA
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Verify:
```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```
Expected: a version number (e.g. `2.11.0+cu128`) and `True`.

### 2.3 — Install all dependencies
```bash
pip install -r ~/TMBL/requirements.txt --upgrade-strategy only-if-needed
```

### 2.4 — Install the CMU MultimodalSDK
```bash
cd ~/TMBL/CMU-MultimodalSDK-main
pip install setuptools
pip install -e . --no-build-isolation
```

Verify:
```bash
python3 -c "from mmsdk import mmdatasdk as md; print('OK')"
```
Expected: `OK`

---

## Section 3 — Clone the Code (one-time only)

```bash
git clone https://github.com/JackAILab/TMBL.git ~/TMBL
mkdir -p ~/TMBL/checkpoints/devBest
mkdir -p ~/data/MOSEI
```

---

## Section 4 — Set Up the CMU-MOSEI Data (one-time only)

The CMU data servers are unreliable. Download the dataset from Kaggle instead and place the files manually.

### 4.1 — Find the unzipped .csd files

```bash
find ~/data/mosei_kaggle -name "*.csd"
```

This shows you where the files landed after unzipping. Note the folder path.

### 4.2 — Copy and rename the 7 required files into `~/data/MOSEI/`

The code expects **exact** filenames. Copy each file, renaming where needed:

```bash
# The 4 files with correct names already — just copy:
cp <path>/CMU_MOSEI_TimestampedWords.csd   ~/data/MOSEI/CMU_MOSEI_TimestampedWords.csd
cp <path>/CMU_MOSEI_Labels.csd             ~/data/MOSEI/CMU_MOSEI_Labels.csd
cp <path>/CMU_MOSEI_TimestampedPhones.csd  ~/data/MOSEI/CMU_MOSEI_TimestampedPhones.csd
cp <path>/CMU_MOSEI_VisualFacet42.csd      ~/data/MOSEI/CMU_MOSEI_VisualFacet42.csd

# The 3 files with typos in the Kaggle dataset — copy AND rename:
cp <path>/CMU_MOSIE_CAVAREP.csd            ~/data/MOSEI/CMU_MOSEI_COVAREP.csd
cp <path>/CMU_MOSEI_TimestampedWordvectors.csd  ~/data/MOSEI/CMU_MOSEI_TimestampedWordVectors.csd
cp <path>/CMU_MOSEI_VisualOpenFace2csd     ~/data/MOSEI/CMU_MOSEI_VisualOpenFace2.csd
```

Replace `<path>` with the folder path found in step 4.1.

Verify all 7 are in place:
```bash
ls ~/data/MOSEI/
```

You should see all 7 `.csd` files listed.

### 4.3 — Create a dummy GloVe file

The code calls `load_emb()` which reads a GloVe word embedding text file. However, the active model (AOTN) uses BERT for text encoding and does **not** actually use GloVe embeddings at runtime. Creating an empty placeholder file is sufficient — the code will create a random embedding matrix, save it as `embedding_and_mapping.pt`, and training will work normally.

```bash
touch ~/data/MOSEI/glove_dummy.txt
```

---

## Section 5 — Fix the Code (one-time only)

Two changes are needed before the code will run correctly on this machine.

### 5.1 — Fix the import in `solver.py`

The original code references the developer's local folder structure. Open `~/TMBL/solver.py` and change line 6:

**Before:**
```python
import MultimodalSA.network as new_models
```

**After:**
```python
import network as new_models
```

Using nano:
```bash
nano ~/TMBL/solver.py
```
Make the change, then save: `Ctrl+O` → Enter → `Ctrl+X`

### 5.2 — Update paths in `config.py`

Open `~/TMBL/config.py` (`nano ~/TMBL/config.py`) and update these lines:

```python
word_emb_path = '/home/elangoa/data/MOSEI/glove_dummy.txt'

project_dir = "/home/elangoa/data/"
sdk_dir = '/home/elangoa/TMBL/CMU-MultimodalSDK-main'
data_dir = os.path.join(project_dir, 'Multimodal')
data_dict = {
    'mosi':     os.path.join(data_dir, 'MOSI'),
    'mosei':    '/home/elangoa/data/MOSEI',
    'ur_funny': os.path.join(data_dir, 'UR_FUNNY')
}
```

### 5.3 — Update paths in `solver.py`

In the same `solver.py` file, update these two lines near the top:

```python
checkpoint_path = "/home/elangoa/TMBL/checkpoints/"
log_file = "/home/elangoa/TMBL/checkpoints/log.txt"
```

---

## Section 6 — Run the Model

### Run in foreground first (to confirm everything works)

```bash
source ~/tmbl_env/bin/activate
cd ~/TMBL
python3 train.py
```

**What to watch for on first run:**
1. Prints `mosei` and the config settings
2. Detects no `.pkl` files → starts processing the `.csd` files into pkl format (takes some minutes)
3. Saves `train.pkl`, `dev.pkl`, `test.pkl`, `embedding_and_mapping.pt` to `~/data/MOSEI/`
4. Training begins — you see `Epoch 0 - Training loss: ...`

Once you confirm Epoch 0 is printing correctly, press `Ctrl+C` and restart in background.

### Run in background (for long training runs)

```bash
nohup python3 train.py > ~/TMBL/run.log 2>&1 &
echo "Training started, PID: $!"
```

Watch live output:
```bash
tail -f ~/TMBL/run.log
```

Stop watching: `Ctrl+C` (does **not** stop training).

Check if training is still running:
```bash
ps aux | grep train.py
```

Stop training:
```bash
kill <PID>
```

---

## What to Expect During Training

Each epoch prints:
```
Epoch 0 - Training loss: X.XXXX  Time: ...
val_mult_acc7: X.XXXX
Accuracy_val (pos/neg): X.XXXX
test_mult_acc7: X.XXXX
mae: X.XXXX
corr: X.XXXX
F1 (pos/neg): X.XXXX
```

Best model checkpoints are saved automatically to `~/TMBL/checkpoints/devBest/`.

---

## Quick Checklist

- [ ] SSH connected and `source ~/tmbl_env/bin/activate` run
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] All 7 `.csd` files present in `~/data/MOSEI/` with correct names
- [ ] `~/data/MOSEI/glove_dummy.txt` exists (empty file)
- [ ] `solver.py` line 6: `import network as new_models`
- [ ] `config.py` paths updated to `/home/elangoa/...`
- [ ] `solver.py` checkpoint and log paths updated
- [ ] `~/TMBL/checkpoints/devBest/` exists

---

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'mmsdk'` | `cd ~/TMBL/CMU-MultimodalSDK-main && pip install -e . --no-build-isolation` |
| `ModuleNotFoundError: No module named 'MultimodalSA'` | Change line 6 of `solver.py` to `import network as new_models` |
| `ModuleNotFoundError: No module named 'pkg_resources'` | `pip install setuptools` then retry mmsdk install |
| `ModuleNotFoundError: No module named 'requests'` | `pip install requests` |
| `ConnectTimeoutError: immortal.multicomp.cs.cmu.edu` | CMU servers are down — ensure all 7 `.csd` files are in `~/data/MOSEI/` so the download is skipped |
| `FileNotFoundError` for `.csd` file | Check `ls ~/data/MOSEI/` — a file is missing or misnamed |
| `FileNotFoundError` for GloVe file | Run `touch ~/data/MOSEI/glove_dummy.txt` and confirm `word_emb_path` in `config.py` points to it |
| `CUDA out of memory` | Reduce `batch_size` in `config.py` (try 32 instead of 48) |
| `torch.cuda.is_available()` returns `False` | `pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall` |
| SSH session drops mid-training | Use `nohup ... &` to run in background — training survives disconnects |
