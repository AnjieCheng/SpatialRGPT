## OpenSpatialDataset Synthesis Pipeline

### Installation

#### Environment

```sh
conda create -n osd_pipeline anaconda python=3.10
conda activate osd_pipeline

##### Install Pytorch according to your own setup #####
# For example, if you have a GPU with CUDA 12.1
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
# This is optional if you prefer to system built-in nvcc.
conda install -c nvidia cuda-toolkit -y

# We use mmengine for config management
pip install -U openmim
mim install mmengine

# Install Wis3D for visualization
pip install https://github.com/zju3dv/Wis3D/releases/download/2.0.0/wis3d-2.0.0-py3-none-any.whl

# Install detectron2 for SOM visualization
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Some other libraries
pip install iopath pyequilib==0.3.0 albumentations einops
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html

```

#### Install Grounded-SAM package

```
mkdir external && cd osdsynth/external
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
```

Follow the instructions on the original [repo](https://github.com/IDEA-Research/Grounded-Segment-Anything#install-without-docker). Our pipeline has been tested with the codebase at this [commit](https://github.com/open-mmlab/mmengine/commit/85c83ba61689907fb1775713622b1b146d82277b). Grounded-SAM codebase at later commits may require some adaptations. If you encounter problems installing the RAM package, try upgrade your `setuptools` version to the latest version.

#### Install Perspective Fields package

```sh
# Skip this if you are already in the `external` dir
cd osdsynth/external
git clone https://github.com/jinlinyi/PerspectiveFields.git
```

#### Download Weights

```sh
cd ../ # navigate back to dataset_pipeline folder
sh ./scripts/download_all_weights.sh
```

### Inference

To specify the folder containing the images for testing, use the `--input` argument. You can also adjust the settings in `configs/v2.py` to better suit your images, like modifying the SAM thresholds or tweaking the DBSCAN hyperparameters.

```sh
python run.py --config configs/v2.py --input PATH_TO_INPUT --vis
```

The results are saved in two formats. One is in JSON, where the Open3D bounding boxes are serialized. If you'd like to recreate the Open3D bounding box object for each detection, you can use the following code:

```python
bbox = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=bbox_dict["min_bound"],
    max_bound=bbox_dict["max_bound"]
)
```

The other format is compatible with Wis3D point clouds. You can use the instructions below to visualize these results.

### Wis3D Visualization

```sh
wis3d --vis_dir ./demo_out/log/Wis3D --host 0.0.0.0 --port 19090
```

You should be able to see the output like this:

<p align="center">
  <img src="./asssets/wis3d-demo.gif" alt="Wis3D Demo">
</p>
