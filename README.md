# SpatialVLA with LIBERO

This is the evaluation project of LIBERO for the [spatialvla](https://github.com/SpatialVLA/SpatialVLA) without fine-tuning, yet there are instances where the robot fails to grasp or shakes uncontrollably, leading to an anticipated absolute success rate of 0. ([rollouts/test.mp4](rollouts/test.mp4))

NOTE: unnorm_key = 'droid/1.0.0' with franka robot
```bash
# Launch LIBERO-Spatial evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family svla\
  --pretrained_checkpoint /root/autodl-tmp/spatialvla-4b-224-pt \
  --task_suite_name libero_spatial \
  --center_crop True

# Launch LIBERO-Object evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family svla\
  --pretrained_checkpoint /root/autodl-tmp/spatialvla-4b-224-pt \
  --task_suite_name libero_object \
  --center_crop True

# Launch LIBERO-Goal evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family svla\
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --center_crop True

# Launch LIBERO-10 (LIBERO-Long) evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family svla\
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True

```


<div align="center">


# SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Models
A spatial-enhanced vision-language-action model trained on 1.1 Million real robot episodes. 🤗
purely huggingFace-based, concise code with efficient performance.

<!-- <div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/930e6814-8a9f-43e1-a284-118a5732daa4">
  <br>
</div> -->

[\[📄Paper\]](https://arxiv.org/pdf/2501.15830)  [\[🔥Project Page\]](https://spatialvla.github.io/) [\[📖 Document\]](#documents) [\[🚀 Quick Start\]](#-quick-start) [\[✅ Performance\]](#-performance-in-simulation-and-real-world) [\[🤗 FAQs\]](#-faqs)

[\[🔥Pre-train\]](#-pre-train-from-scratch) [\[🚀 Fine-tune\]](#-fine-tune-from-spatialvla) [\[🎄Custom Dataset\]](#-use-custom-datasets)

![perform](.assets/teaser.png)

</div>

## News 🚀🚀🚀
- `2025/01/29`: We release the [SpatialVLA 1.0](https://huggingface.co/collections/IPEC-COMMUNITY/foundation-vision-language-action-model-6795eb96a9c661f90236acbb). SpatialVLA achieves state-of-the-art performance across a diverse range of evaluations and shows significantly faster inference speed with fewer tokens per action.
- `2025/02/06`: We release the SimplerEnv evaluation code for SpatialVLA. Please refer to [DelinQu/SimplerEnv-OpenVLA](https://github.com/DelinQu/SimplerEnv-OpenVLA/), and make sure `transformers >= 4.47.0`.

## Documents

### 🚀 Quick Start

SpatialVLA relies solely on HuggingFace Transformers 🤗, making deployment extremely easy. If your environment supports `transformers >= 4.47.0`, you can directly use the following code to load the model and perform inference. (requires 8.5GB of GPU memory).

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

model_name_or_path="IPEC-COMMUNITY/spatialvla-4b-224-pt"
processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16).eval().cuda()

image = Image.open("example.png").convert("RGB")
prompt = "What action should the robot take to pick the cup?"
inputs = processor(images=[image], text=prompt, return_tensors="pt")
generation_outputs = model.predict_action(inputs)

actions = processor.decode_actions(generation_outputs, unnorm_key="bridge_orig/1.0.0")
print(actions)
```

If you want to use the model for fine-tuning or pre-training, you need to install the required packages and download the model from the Hugging Face model hub. The VLM backbone of SpatialVLA is PaLiGemma2, which requires transformers >= 4.47.0. Hence, create a Python environment with Python >= 3.10.

```bash
conda create -n spatialvla python=3.10
conda activate spatialvla
```

Install packages from `requirements.txt` file. Note that we use a customised `dlimp` to support seed setting for reproducibility. If you catch any problems, please manually install the dlimp form the [dlimp_custom](https://github.com/SpatialVLA/dlimp_custom).

```bash
pip install -r requirements.txt
```

### 🌟 **Pre-train from Scratch**
SpatialVLA is pre-trained with 1.1 Million real-robot demonstrations from the OXE and RH20T dataset on a cluster of 64 A100 GPUs for abut 10 days, using a batch size of 2048. You can pre-train the model from scratch using the following command.

```bash
# torchrun
bash scripts/spatialvla_4b_pretrain/torchrun_pretrain.sh

# or in a slurm cluster
bash scripts/spatialvla_4b_pretrain/slurm_pretrain.sh
```

### 🌟 **Fine-tune from SpatialVLA**

Most of our fine-tuning experiments are conducted using LoRA on 4 or 8 A100 GPUs.
You can use the following scripts for full-parameter or LoRA fine-tuning. For real-world experiments with small datasets, we prefer using LoRA for fine-tuning.

```bash
# full fine-tuning
bash scripts/spatialvla_4b_finetune/finetune_full.sh

# LoRA fine-tuning
bash scripts/spatialvla_4b_finetune/finetune_lora.sh
```


### 🌟 **SimplerEnv Benchmark**
We release the SimplerEnv evaluation code for SpatialVLA based on [DelinQu/SimplerEnv-OpenVLA](https://github.com/DelinQu/SimplerEnv-OpenVLA/). Please install the simpler_env environment by following [DelinQu/SimplerEnv-OpenVLA](https://github.com/DelinQu/SimplerEnv-OpenVLA/) and make sure `transformers >= 4.47.0`. After install all the dependencies, you can perform the evaluation by: 

```bash
# under the project dir of SimplerEnv-OpenVLA/
bash scripts/run_spatialvla.sh
```
Note: Similar to most papers, e.g., HPT and TraceVLA, we omitted the `Open Top Drawer and Place Apple` from our evaluation, since the vast majority of policies achieved scores approaching 0 on this task.

### 🎄 Use Custom Datasets
TODO

## 🤗 Model Zoo

<table>
  <tr>
    <th>Model Name</th>
    <th>VLM Backbone</th>
    <th>VLA Model</th>
  </tr>
  <tr>
    <td>SpatialVLA-4B-224-pt</td>
    <td><a href="https://huggingface.co/google/paligemma2-3b-pt-224">google/paligemma2-3b-pt-224</a></td>
    <td><a href="https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-pt">spatialvla-4b-224-pt</a></td>
  </tr>
  <tr>
    <td>SpatialVLA-4B-mix-224-pt</td>
    <td><a href="https://huggingface.co/google/paligemma2-3b-pt-224">google/paligemma2-3b-pt-224</a></td>
    <td><a href="https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-mix-224-pt">spatialvla-4b-mix-224-pt</a></td>
  </tr>
</table>

## ✅ Performance in Simulation and Real-world
<details>
  <summary>
  SimplerEnv evaluation on Google Robot tasks.
  </summary>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: center;">
      <th rowspan="2">Model</th>
      <th colspan="4">Visual Matching</th>
      <th colspan="4">Variant Aggregation</th>
    </tr>
    <tr style="text-align: center;">
      <th>Pick Coke Can</th>
      <th>Move Near</th>
      <th>Open/Close Drawer</th>
      <th>#Average</th>
      <th>Pick Coke Can</th>
      <th>Move Near</th>
      <th>Open/Close Drawer</th>
      <th>#Average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>RT-1 (Begin)</td>
      <td>2.7%</td>
      <td>5.0%</td>
      <td>13.9%</td>
      <td>6.8%</td>
      <td>2.2%</td>
      <td>4.0%</td>
      <td>6.9%</td>
      <td>4.2%</td>
    </tr>
    <tr>
      <td>RT-1 (15%)</td>
      <td>71.0%</td>
      <td>35.4%</td>
      <td>56.5%</td>
      <td>60.2%</td>
      <td>81.3%</td>
      <td>44.6%</td>
      <td>26.7%</td>
      <td>56.2%</td>
    </tr>
    <tr>
      <td>RT-1 (Converged)</td>
      <td>85.7%</td>
      <td>44.2%</td>
      <td>73.0%</td>
      <td>74.6%</td>
      <td>89.8%</td>
      <td>50.0%</td>
      <td>32.3%</td>
      <td>63.3%</td>
    </tr>
    <tr>
      <td>HPT</td>
      <td>56.0%</td>
      <td>60.0%</td>
      <td>24.0%</td>
      <td>46.0%</td>
      <td>--</td>
      <td>--</td>
      <td>31.0%</td>
      <td>45.0%</td>
    </tr>
    <tr>
      <td>TraceVLA</td>
      <td>28.0%</td>
      <td>53.7%</td>
      <td>57.0%</td>
      <td>42.0%</td>
      <td>60.0%</td>
      <td>56.4%</td>
      <td>29.4%</td>
      <td>39.6%</td>
    </tr>
    <tr>
      <td>RT-1-X</td>
      <td>56.7%</td>
      <td>31.7%</td>
      <td>59.7%</td>
      <td>53.4%</td>
      <td>49.0%</td>
      <td>32.3%</td>
      <td>35.3%</td>
      <td>64.3%</td>
    </tr>
    <tr>
      <td>RT-2-X</td>
      <td>78.7%</td>
      <td>77.9%</td>
      <td>25.0%</td>
      <td>60.7%</td>
      <td>82.3%</td>
      <td>79.2%</td>
      <td>--</td>
      <td>--</td>
    </tr>
  <tr>
      <td>Octo-Base</td>
      <td>17.0%</td>
      <td>4.2%</td>
      <td>22.7%</td>
      <td>16.8%</td>
      <td>0.6%</td>
      <td>3.1%</td>
      <td>1.1%</td>
      <td>1.1%</td>
    </tr>
    <tr>
      <td>OpenVLA</td>
      <td>16.3%</td>
      <td>46.2%</td>
      <td>35.6%</td>
      <td>27.7%</td>
      <td>54.5%</td>
      <td>47.7%</td>
      <td>17.7%</td>
      <td>39.8%</td>
    </tr>
    <tr>
      <td>RoboVLM (zero-shot)</td>
      <td>72.7%</td>
      <td>66.3%</td>
      <td>26.8%</td>
      <td>56.3%</td>
      <td>68.3%</td>
      <td>56.0%</td>
      <td>8.5%</td>
      <td>46.3%</td>
    </tr>
    <tr>
      <td>RoboVLM (fine-tuning)</td>
      <td>77.3%</td>
      <td>61.7%</td>
      <td>43.5%</td>
      <td>63.4%</td>
      <td>75.6%</td>
      <td>60.0%</td>
      <td>10.6%</td>
      <td>51.3%</td>
    </tr>
    <tr>
      <td>SpatialVLA (zero-shot)</td>
      <td><b>81.0%</b></td>
      <td><b>69.6%</b></td>
      <td><b>59.3%</b></td>
      <td><b>71.9%</b></td>
      <td><b>89.5%</b></td>
      <td><b>71.7%</b></td>
      <td>36.2%</td>
      <td><b>68.8%</b></td>
    </tr>
    <tr>
      <td>SpatialVLA (fine-tuning)</td>
      <td><b>86.0%</b></td>
      <td><b>77.9%</b></td>
      <td>57.4%</td>
      <td><b>75.1%</b></td>
      <td>88.0%</td>
      <td>72.7%</td>
      <td>41.8%</td>
      <td><b>70.7%</b></td>
    </tr>
  </tbody>
</table>

</details>


<details>
  <summary>
  SimplerEnv evaluation on WidowX Robot tasks.
  </summary>
  <table border="1" class="dataframe">
    <thead>
      <tr style="text-align: center;">
        <th rowspan="2">Model</th>
        <th colspan="2">Put Spoon on Towel</th>
        <th colspan="2">Put Carrot on Plate</th>
        <th colspan="2">Stack Green Block on Yellow Block</th>
        <th colspan="2">Put Eggplant in Yellow Basket</th>
        <th rowspan="2">#Overall Average</th>
      </tr>
      <tr style="text-align: center;">
        <th>Grasp Spoon</th>
        <th>Success</th>
        <th>Grasp Carrot</th>
        <th>Success</th>
        <th>Grasp Green Block</th>
        <th>Success</th>
        <th>Grasp Eggplant</th>
        <th>Success</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>RT-1-X</td>
        <td>16.7%</td>
        <td>0.0%</td>
        <td>20.8%</td>
        <td>4.2%</td>
        <td>8.3%</td>
        <td>0.0%</td>
        <td>0.0%</td>
        <td>0.0%</td>
        <td>1.1%</td>
      </tr>
      <tr>
        <td>Octo-Base</td>
        <td>34.7%</td>
        <td>12.5%</td>
        <td>52.8%</td>
        <td>8.3%</td>
        <td>31.9%</td>
        <td>0.0%</td>
        <td>66.7%</td>
        <td>43.1%</td>
        <td>16.0%</td>
      </tr>
      <tr>
        <td>Octo-Small</td>
        <td>77.8%</td>
        <td>47.2%</td>
        <td>27.8%</td>
        <td>9.7%</td>
        <td>40.3%</td>
        <td>4.2%</td>
        <td>87.5%</td>
        <td>56.9%</td>
        <td>30.0%</td>
      </tr>
      <tr>
        <td>OpenVLA</td>
        <td>4.1%</td>
        <td>0.0%</td>
        <td>33.3%</td>
        <td>0.0%</td>
        <td>12.5%</td>
        <td>0.0%</td>
        <td>8.3%</td>
        <td>4.1%</td>
        <td>1.0%</td>
      </tr>
      <tr>
        <td>RoboVLM (zero-shot)</td>
        <td>37.5%</td>
        <td>20.8%</td>
        <td>33.3%</td>
        <td>25.0%</td>
        <td>8.3%</td>
        <td>8.3%</td>
        <td>0.0%</td>
        <td>0.0%</td>
        <td>13.5%</td>
      </tr>
      <tr>
        <td>RoboVLM (fine-tuning)</td>
        <td>54.2%</td>
        <td>29.2%</td>
        <td>25.0%</td>
        <td>25.0%</td>
        <td>45.8%</td>
        <td>12.5%</td>
        <td>58.3%</td>
        <td>58.3%</td>
        <td>31.3%</td>
      </tr>
      <tr>
        <td>SpatialVLA (zero-shot)</td>
        <td><b>25.0%</b></td>
        <td><b>20.8%</b></td>
        <td><b>41.7%</b></td>
        <td>20.8%</td>
        <td><b>58.3%</b></td>
        <td>25.0%</td>
        <td><b>79.2%</b></td>
        <td>70.8%</td>
        <td><b>34.4%</b></td>
      </tr>
      <tr>
        <td>SpatialVLA (fine-tuning)</td>
        <td><b>20.8%</b></td>
        <td>16.7%</td>
        <td>29.2%</td>
        <td>25.0%</td>
        <td><b>62.5%</b></td>
        <td>29.2%</td>
        <td><b>100.0%</b></td>
        <td><b>100.0%</b></td>
        <td><b>42.7%</b></td>
      </tr>
    </tbody>
  </table>
</details>

<details>
  <summary>LIBERO Simulation Benchmark Results.</summary>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: center;">
      <th rowspan="2">Model</th>
      <th colspan="2">LIBERO-Spatial</th>
      <th colspan="2">LIBERO-Object</th>
      <th colspan="2">LIBERO-Goal</th>
      <th colspan="2">LIBERO-Long</th>
      <th colspan="2">Average</th>
    </tr>
    <tr style="text-align: center;">
      <th>SR (↑)</th>
      <th>Rank (↓)</th>
      <th>SR (↑)</th>
      <th>Rank (↓)</th>
      <th>SR (↑)</th>
      <th>Rank (↓)</th>
      <th>SR (↑)</th>
      <th>Rank (↓)</th>
      <th>SR (↑)</th>
      <th>Rank (↓)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Diffusion Policy from scratch</td>
      <td>78.3 ± 1.1%</td>
      <td>5</td>
      <td><b>92.5 ± 0.7%</b></td>
      <td>1</td>
      <td>68.3 ± 1.2%</td>
      <td>5</td>
      <td>50.5 ± 1.3%</td>
      <td>5</td>
      <td>72.4 ± 0.7%</td>
      <td>5</td>
    </tr>
    <tr>
      <td>Octo fine-tuned</td>
      <td>78.9 ± 1.0%</td>
      <td>4</td>
      <td>85.7 ± 0.9%</td>
      <td>4</td>
      <td><b>84.6 ± 0.9%</b></td>
      <td>1</td>
      <td>51.1 ± 1.3%</td>
      <td>4</td>
      <td>75.1 ± 0.6%</td>
      <td>3</td>
    </tr>
    <tr>
      <td>OpenVLA fine-tuned</td>
      <td>84.7 ± 0.9%</td>
      <td>2</td>
      <td>88.4 ± 0.8%</td>
      <td>3</td>
      <td>79.2 ± 1.0%</td>
      <td>2</td>
      <td>53.7 ± 1.3%</td>
      <td>3</td>
      <td>76.5 ± 0.6%</td>
      <td>2</td>
    </tr>
    <tr>
      <td>TraceVLA fine-tuned</td>
      <td>84.6 ± 0.2%</td>
      <td>3</td>
      <td>85.2 ± 0.4%</td>
      <td>5</td>
      <td>75.1 ± 0.3%</td>
      <td>4</td>
      <td>54.1 ± 1.0%</td>
      <td>2</td>
      <td>74.8 ± 0.5%</td>
      <td>4</td>
    </tr>
    <tr>
      <td>SpatialVLA fine-tuned</td>
      <td><b>88.2 ± 0.5%</b></td>
      <td>1</td>
      <td>89.9 ± 0.7%</td>
      <td>2</td>
      <td>78.6 ± 0.6%</td>
      <td>3</td>
      <td><b>55.5 ± 1.0%</b></td>
      <td>1</td>
      <td><b>78.1 ± 0.7%</b></td>
      <td>1</td>
    </tr>
  </tbody>
</table>

</details>

<details>
  <summary>Zero-shot Robot Control Evaluation on WidowX Robot.</summary>
  <img src=".assets/widowX_zeroshot.png" alt="perform">
</details>

<details>
  <summary>Spatial Understanding Capability Evaluation..</summary>
  <img src=".assets/spatial_setup.png" alt="perform">
</details>

<details>
  <summary>Adapting to New Robot Setups on Franka Robot.</summary>
  <img src=".assets/franka_sft.png" alt="perform">
</details>

## TODO List

- [x] Release pre-training / fine-tuning code for SpatialVLA series.
- [x] Release the code, model, and custom data of SpatialVLA.
- [x] Release the SimplerENV evaluation code for SpatialVLA series
- [ ] Release SpatialVLA2

## 🤗 FAQs
If you encounter any issues, feel free to open an issue on GitHub or reach out through discussions. We appreciate your feedback and contributions! 🚀

## License

This project is released under the [MIT license](LICENSE). Parts of this project contain code and models from other sources, which are subject to their respective licenses.

## Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@misc{qu2025spatialvlaexploringspatialrepresentations,
      title={SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model}, 
      author={Delin Qu and Haoming Song and Qizhi Chen and Yuanqi Yao and Xinyi Ye and Yan Ding and Zhigang Wang and JiaYuan Gu and Bin Zhao and Dong Wang and Xuelong Li},
      year={2025},
      eprint={2501.15830},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2501.15830}, 
}
```

## Acknowledgement
SpatialVLA is built with reference to the code of the following projects: [InternVL](https://github.com/OpenGVLab/InternVL), [Google Paligemma2](https://huggingface.co/google/paligemma2-3b-pt-224), [Transformers](https://github.com/huggingface/transformers), [OpenVLA](https://github.com/openvla/openvla) and [ZoeDepth](https://huggingface.co/spaces/shariqfarooq/ZoeDepth). Thanks for their awesome work!
