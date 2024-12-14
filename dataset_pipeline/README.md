## OpenSpatialDataset Synthesis Pipeline

### Installation

#### Environment

```sh
conda create -n osd_pipeline python=3.10 -y
conda activate osd_pipeline

##### Install Pytorch according to your own setup #####
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# We use mmengine for config management
pip install -U openmim
mim install mmengine

# Install Wis3D for visualization
pip install https://github.com/zju3dv/Wis3D/releases/download/2.0.0/wis3d-2.0.0-py3-none-any.whl

# Install detectron2 for SOM visualization
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Some other libraries
pip install iopath pyequilib==0.3.0 albumentations einops open3d imageio
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html
```

#### Install Grounded-SAM package

```sh
mkdir osdsynth/external && cd osdsynth/external
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
```

Follow the instructions on the original [repo](https://github.com/IDEA-Research/Grounded-Segment-Anything#install-without-docker) to build Segment Anything, Grounding DINO, and RAM, respectively. Our pipeline has been tested with the codebase at this [commit](https://github.com/IDEA-Research/Grounded-Segment-Anything/tree/126abe633ffe333e16e4a0a4e946bc1003caf757).

```sh
cd Grounded-Segment-Anything/

# Install Segment Anything
python -m pip install -e segment_anything

# Install Grounding DINO
pip install --no-build-isolation -e GroundingDINO

# Install RAM
git clone https://github.com/xinyu1205/recognize-anything.git
pip install -r ./recognize-anything/requirements.txt
pip install setuptools --upgrade
pip install -e ./recognize-anything/
```


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

#### Template-based QA
To specify the folder containing the images for testing, use the `--input` argument. You can also adjust the settings in `configs/v2.py` to better suit your images, like modifying the SAM thresholds or tweaking the DBSCAN hyperparameters.

```sh
python run_template_qa.py --config configs/v2.py --input PATH_TO_INPUT --vis True
```

The results are saved in two formats. One is in JSON, where the Open3D bounding boxes are serialized. If you'd like to recreate the Open3D bounding box object for each detection, you can use the following code:

```python
bbox = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=bbox_dict["min_bound"],
    max_bound=bbox_dict["max_bound"]
)
```

The other format is compatible with Wis3D point clouds. You can use the instructions below to visualize these results.


#### LLM-rephrased QA

**Step1:** Generate template-based descriptions with the following command, this will save a `llm_prompts.json` in the output json folder.

```sh
python run_template_facts.py --config configs/v2.py --input PATH_TO_INPUT --vis True
```

**Step2:** Prepare a clean environment and install sglang
```sh
conda create -n sglang python=3.10 -y
conda activate sglang

pip install --upgrade pip
pip install "sglang[all]"

pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
```

**Step3:** Run llm rephrase, currently the script is using Llama-3.1-70B
```sh
export HF_TOKEN=<key>
python run_llm.py --llm-prompts-path /PATH/SAMPLE_llm_prompts.json --port 3000 --gpus 8

```

### Wis3D Visualization

```sh
wis3d --vis_dir ./demo_out/log/Wis3D --host 0.0.0.0 --port 19090
```

You should be able to see the output like this:

<p align="center">
  <img src="./asssets/wis3d-demo.gif" alt="Wis3D Demo">
</p>
