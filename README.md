
# SANPO: (S)cene understanding for (A)ccessible (N)avigation and (P)athfinding (O)utdoors Dataset

<div align="center">

<p align="center">
<img src="res/sanpo.gif" width="800px">
</p>

<p align="center">
<a href="#dataset"><b>Dataset</b></a> •
<a href="https://arxiv.org/pdf/2103.03375.pdf"><b>Paper</b></a> •
<a href="#download-data"><b>Download Data</b></a> •
<a href="#license--contact"><b>License & Contact</b></a>
</p>

</div>

## Dataset
**S**cene understanding for **A**ccessible **N**avigation and **P**athfinding
 **O**utdoors is a multi-attribute dataset of common outdoor scenes from urban,
park, and suburban settings. At its core, SANPO is a video-first dataset.
It has both real (SANPO-Real) and synthetic (SANPO-Synthetic) counterparts.
The real data is collected via an extensive data collection effort.
The synthetic data is curated in collaboration with our external partner,
[Parallel Domain](https://paralleldomain.com/).

### Salient Features
* Multi attribute stereo video dataset
* Real as well as synthetic data
* Depth & odometry labels
* Temporally consistent segmentation annotations
* High level attributes like environment type, visibility, motion etc.

### Dataset Contents
Each sample in the dataset is denoted by as a **session**.

A **SANPO-Real session** contains:

- High level session attributes like environment type, visibility etc.
- Two stereo videos
- Cameras' hardware information
- IMU data
- One depth map (meters) and one disparity map (pixels) for each of the stereo videos (wrt to left side)
- Optional temporally consistent video segmentation annotation (wrt left side)

A **SANPO-Synthetic session** contains:

- One video
- Camera's hardware information used in the simulation
- IMU data
- Depth map (in meters)
- Temporally consistent video segmentation annotation

All the video data is in PNG format.
Segmentation masks are saved as PNG files as well.
Depth maps are in numpy arrays (saved as npz files).
All other relevant data
(including segmentation taxonomy, IMU, session attributes)
is either in csv or json files.

### Train/Test Splits
We provide lists of mutually exclusive session IDs for training and testing
 for both the real and synthetic counterparts of our dataset.

### Privacy
Privacy has been our highest priority from the
dataset's inception. All the videos are processed through PII removal to blur faces and license plates. If any sample is found to be inadequately processed,
please contact us immediately at <a href="mailto:sanpo_dataset@google.com">sanpo_dataset@google.com</a>.

## Paper
**(TODO: Add link to paper or blogpost.)**

## Download Data
**(TODO: Upload data to gcp and activate the link.)**

All SANPO data can be downloaded directly from our [Google Cloud Storage bucket](https://console.cloud.google.com/storage/browser/sanpo_dataset).
You can also browse through the dataset and download specific files using the `gsutil cp` command:
```
gsutil -m cp -r "gs://sanpo_dataset/sanpo_dataset/{FILE_OR_DIR_PATH}" .
```
See [here](https://cloud.google.com/storage/docs/gsutil) for instructions on installing the `gsutil` tool.

## License & Contact
We release SANPO dataset under the <a href="https://creativecommons.org/licenses/by/4.0/">Creative Commons V4.0</a> license. You are free to share and adapt this data for any purpose. If you found this dataset useful, please consider citing our paper **(TODO: Add paper's link)**.

```
TODO: Add paper's bibtex.
```

If you have any questions about the dataset or paper, please send us an email at <a href="mailto:sanpo_dataset@google.com">sanpo_dataset@google.com</a>.
