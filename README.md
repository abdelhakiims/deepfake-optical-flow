# Deepfake video detection with motion-compensated optical flow

A deepfake video detector that looks at **how a face moves**, not only at how it looks. The head's
rigid motion is removed from the optical flow, so that what remains is the motion *inside* the face,
where face swaps leave inconsistencies. A two-stream network (optical flow + face image, ResNet18
each) reads short clips of consecutive frames, and an LSTM combines them over time.

```
video -> face tracking -> clips of 21 consecutive face crops -> RAFT optical flow
      -> head-motion removal (affine fit per flow map)
      -> [ResNet18 on flow | ResNet18 on face image] per frame -> LSTM -> real / fake
```

Everything that was tested, every design decision and its reason, every result (good and bad), and
the problems met along the way are documented in **[EXPERIMENTS.md](EXPERIMENTS.md)** (French version:
[EXPERIMENTS_fr.md](EXPERIMENTS_fr.md); Word and PDF versions of both in `docs/`).

## Results so far

Video-level AUC. "Unseen" = a dataset never used in training.

| Trained on | Tested on | AUC | 95% interval |
|---|---|---|---|
| DFD (7-fold by actor) | DFD, unseen actors | 0.972 ± 0.017 | over 7 folds |
| DFD | Celeb-DF (unseen) | 0.781 ± 0.032 | over 7 models |
| FaceForensics++ (official split) | FaceForensics++ test | 0.967 | 0.955–0.977 |
| FaceForensics++ | DFD (unseen) | 0.820 | 0.799–0.842 |
| FaceForensics++ | Celeb-DF (unseen) | 0.719 | 0.672–0.764 |
| FaceForensics++, mean motion removal | DFD / Celeb-DF (unseen) | 0.870 / 0.739 | 0.850–0.891 / 0.691–0.786 |

On unseen datasets this is currently below published detectors (about 0.80–0.91 on Celeb-DF when
trained on FaceForensics++). The ablations show that removing only the mean displacement of the head
generalises better than the full affine removal, and that optical flow alone hardly transfers across
datasets; see EXPERIMENTS.md, section 10, for all six versions, the comparison with published methods
and the likely reasons. Each version was trained once (one seed).

## Repository layout

```
deepfake_flow/          the package
  paths.py              where RAFT and the face model are found (arguments, environment, defaults)
  faces.py              face detection (YuNet), tracking, main subject, clip placement
  flow.py               RAFT optical flow; head-motion removal, flip, flow scale
  preprocess.py         video -> face clips -> flow, saved as .npy           [command]
  data.py               preprocessed videos, identity-safe splits, PyTorch dataset
  metrics.py            clip/video metrics, threshold choice, bootstrap interval
  model.py              two-stream ResNet18 + LSTM
  train.py              training, then test on held-out videos              [command]
  evaluate.py           test a trained model on another dataset             [command]
  predict.py            verdict for one video file                          [command]
  summarize.py          one table of all results                            [command]
  experiments.py        every experiment of the project, in order           [command]
  check_env.py          checks Python, packages, GPU, RAFT, face model      [command]
app.py                  web interface (Streamlit)
splits/ffpp/            official FaceForensics++ train/val/test split
tests/                  tests on small random data (CPU only)
models/                 put the face-detection model here (see models/README.md)
third_party/            put the RAFT code here (see third_party/README.md)
EXPERIMENTS.md          design decisions, experiments and results
EXPERIMENTS_fr.md       the same document in French
```

Commands are run from the repository root as `python -m deepfake_flow.<command> --help`.

## Installation

Python 3.10–3.14 and an NVIDIA GPU are needed. The results above were produced on a laptop with an
Intel Core 5 210H processor, 16 GB of DDR5 memory and an NVIDIA GeForce RTX 4050 Laptop GPU (6 GB).

```
python -m venv .venv
.venv\Scripts\activate            (Windows)   or   source .venv/bin/activate   (Linux/macOS)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
git clone https://github.com/princeton-vl/RAFT third_party/RAFT
```

Then download the RAFT weights (`third_party/README.md`) and the YuNet face model (`models/README.md`),
and check the setup:

```
python -m deepfake_flow.check_env
python -m pytest tests
```

## Data

The datasets must be requested from their authors and are not included:

| Dataset | Where to request it | Used for |
|---|---|---|
| FaceForensics++ (c23) and DFD | https://github.com/ondyari/FaceForensics | training (FaceForensics++), training and unseen test (DFD) |
| Celeb-DF v2 | https://github.com/yuezunli/celeb-deepfakeforensics | unseen test (official 518-video test list) |

Keep each dataset in its official folder layout.

## Usage

**Preprocess a folder of videos** (one class at a time):

```
python -m deepfake_flow.preprocess --videos_dir <Celeb-DF-v2>/Celeb-real --label Real --out_root <flows>/CelebDF
```

**Train and test** (here: 7-fold by actor on DFD, fold 0; or the official FaceForensics++ split):

```
python -m deepfake_flow.train --flow_root <flows>/DFD --folds 7 --fold 0 --out_dir runs/dfd_f0
python -m deepfake_flow.train --flow_root <flows>/FFPP --ffpp_splits splits/ffpp --out_dir runs/ffpp
```

Options for the ablations: `--mode two_stream|flow|rgb`, `--motion affine|mean|none`,
`--temporal lstm|mean`.

**Test on another dataset:**

```
python -m deepfake_flow.evaluate --checkpoint runs/ffpp/best.pth --flow_root <flows>/CelebDF \
    --test_list <Celeb-DF-v2>/List_of_testing_videos.txt --out_dir runs/ffpp_to_celebdf
```

**Analyse one video**, on the command line or in the browser:

```
python -m deepfake_flow.predict --video my_video.mp4 --checkpoint runs/ffpp/best.pth
streamlit run app.py -- --checkpoint runs/ffpp/best.pth
```

**Reproduce every experiment** of EXPERIMENTS.md (each stage skips what is already done):

```
python -m deepfake_flow.experiments preprocess-dfd preprocess-celebdf preprocess-ffpp \
    --flows_dir <flows> --dfd_dir <DFD> --celebdf_dir <Celeb-DF-v2> --ffpp_dir <FaceForensics++>
python -m deepfake_flow.experiments dfd-cv dfd-to-celebdf ffpp summary \
    --flows_dir <flows> --celebdf_dir <Celeb-DF-v2>
```

## Evaluation rules

- Videos are split, never clips; no identity appears in two parts of a split (checked in code).
- DFD: 7-fold cross-validation grouped by actor. FaceForensics++: official split. Celeb-DF: official
  test list.
- Model selection and the decision threshold use validation videos only.
- Reported: AUC (with a 95% bootstrap interval), accuracy, balanced accuracy, false-alarm rate, at
  clip and video level. Cross-dataset results are compared by AUC.

## Third-party components

- RAFT optical flow: Z. Teed and J. Deng, "RAFT: Recurrent All-Pairs Field Transforms for Optical
  Flow", ECCV 2020. Code: https://github.com/princeton-vl/RAFT (BSD-3-Clause).
- YuNet face detector, from the OpenCV Zoo: https://github.com/opencv/opencv_zoo.
- ResNet18 with ImageNet weights, from torchvision.
- FaceForensics++ split files: https://github.com/ondyari/FaceForensics.

## License

The code of this repository is released under the MIT License (see [LICENSE](LICENSE)). The datasets,
the RAFT code and weights and the YuNet model are not included and keep their own licenses and terms
of use; the datasets must be requested from their authors.
