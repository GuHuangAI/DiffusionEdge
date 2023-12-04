## DiffusionEdge: Diffusion Probabilistic Model for Crisp Edge Detection  ([arxiv]())
[Yunfan Ye](https://yunfan1202.github.io), [Yuhang Huang](https://github.com/GuHuangAI), [Renjiao Yi](https://renjiaoyi.github.io/), [Zhiping Cai](), [Kai Xu](http://kevinkaixu.net/index.html).

![Teaser](assets/teaser.png)
![](assets/denoising_process/3063/test.gif)
![](assets/denoising_process/5096/test.gif)

# News
- Upload pretrained weights and visualizations of
  [BSDS](https://github.com/GuHuangAI/DiffusionEdge/releases/download/v1/results_bsds_stride240_step5.zip),
  [BIPED](https://github.com/GuHuangAI/DiffusionEdge/releases/download/v1/results_biped_stride240_step5.zip) and
  [NYUD](https://github.com/GuHuangAI/DiffusionEdge/releases/download/v1/results_nyud_stride240_step5.zip).
- We now update a simple demo, please see [Quickly Demo](#iii-quickly-demo-)
- First Committed. 

## I. Before Starting.
1. install torch
~~~
create -n diffedge python=3.9
conda avticate diffedge
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
~~~
2. install other packages.
~~~
pip install -r requirement.txt
~~~
3. prepare accelerate config.
~~~
accelerate config
~~~

## II. Prepare Data.
The training data structure should look like:
```commandline
|-- $data_root
|   |-- image
|   |-- |-- raw
|   |-- |-- |-- XXXXX.jpg
|   |-- |-- |-- XXXXX.jpg
|   |-- edge
|   |-- |-- raw
|   |-- |-- |-- XXXXX.png
|   |-- |-- |-- XXXXX.png
```
The testing data structure should look like:
```commandline
|-- $data_root
|   |-- XXXXX.jpg
|   |-- XXXXX.jpg
```

## III. Quickly Demo !
1. download the pretrained weights:  

| Dataset | ODS (<font color=blue>SEval</font>/<font color=green>CEval</font>) | OIS (<font color=blue>SEval</font>/<font color=green>CEval</font>) | AC    | Weight                                                                               |
|---------|--------------------------------------------------------------------|--------------------------------------------------------------------|-------|--------------------------------------------------------------------------------------|
| BSDS    | <font color=blue>0.834</font> / <font color=green>0.749</font>     | <font color=blue>0.848</font> / <font color=green>0.754</font>     | 0.476 | [download](https://github.com/GuHuangAI/DiffusionEdge/releases/download/v1/bsds.pt)  |
| NYUD    | <font color=blue>0.761</font> / <font color=green>0.732</font>     | <font color=blue>0.766</font> / <font color=green>0.738</font>     | 0.846 | [download](https://github.com/GuHuangAI/DiffusionEdge/releases/download/v1/biped.pt) |\
| BIPED   | <font color=blue>0.899</font>                                      | <font color=blue>0.901</font>                                      | 0.849 | [download](https://github.com/GuHuangAI/DiffusionEdge/releases/download/v1/nyud.pt)  |

2. put your images in a directory and run:
~~~
python demo.py --input_dir $your input dir$ --pre_weight $the downloaded weight path$ --out_dir $the path saves your results$
~~~

## IV. Training.
1. train the first stage model (AutoEncoder):
~~~
accelerate launch train_vae.py --cfg ./configs/first_stage_d4.yaml
~~~
2. you should add the final model weight of the first stage to the config file `./configs/BSDS_train.yaml` (**line 42**), then train latent diffusion-edge model:
~~~
accelerate launch train_cond_ldm.py --cfg ./configs/BSDS_train.yaml
~~~

## V. Inference.
make sure your model weight path is added in the config file `./configs/BSDE_sample.yaml` (**line 73**), and run:
~~~
python sample_cond_ldm.py --cfg ./configs/BSDS_sample.yaml
~~~
Note that you can modify the `sampling_timesteps` (**line 13**) to control the inference speed.

## Concat
If you have some questions, please concat with huangai@nudt.edu.cn.
## Thanks
Thanks to the base code [DDM-Public](https://github.com/GuHuangAI/DDM-Public).
## Citation
~~~
waiting for updating
~~~