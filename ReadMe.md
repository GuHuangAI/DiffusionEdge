## DiffusionEdge: Diffusion Probabilistic Model for Crisp Edge Detection  ([arxiv]())
![Teaser](assets/teaser.png)

## Framework
![Framework](assets/framework.png)

# News
- First Committed.

## I. Before Starting.
1. install torch
~~~
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
The data structure should look like:
```commandline
|-- $data_root
|   |-- image
|   |-- |-- raw
|   |-- |-- |-- real
|   |-- |-- |-- |-- XXXXX.jpg
|   |-- |-- |-- |-- XXXXX.jpg
|   |-- edge
|   |-- |-- raw
|   |-- |-- |-- real
|   |-- |-- |-- |-- XXXXX.png
|   |-- |-- |-- |-- XXXXX.png
```

## III. Quickly Demo !
1. download the pretrained weights:  

| Dataset | ODS   | OIS   | AC    | Weight       |
|---------|-------|-------|-------|--------------|
| BSDS    | 0.834 | 0.848 | 0.476 | [download]() |
| BIPED   | 0.899 | 0.901 | 0.849 | [download]() |
| NYUD    | 0.761 | 0.766 | 0.846 | [download]() |

2. put your images in a directory and run:
~~~
python demo.py --input_dir $your input dir$ --pre_weight $the downloaded weight path$ --out_dir $the path saves your results$
~~~

## IV. Training.
1. training the first stage model (AutoEncoder):
~~~
accelerate launch train_vae.py --cfg ./configs/first_stage_d4.yaml
~~~
2. you should add the final model weight of the first stage to the config file `./configs/BSDS_swin_randresize.yaml` (**line 42**), then train latent diffusion-edge model:
~~~
accelerate launch train_cond_ldm.py --cfg ./configs/BSDS_swin_randresize.yaml
~~~

## V. Inference.
make sure your model weight path is added in the config file, and run:
~~~
python sample_cond_ldm.py --cfg ./configs/BSDS_swin_randresize_sample.yaml
~~~

## Concat
If you have some questions, please concat with huangai@nudt.edu.cn.
## Thanks
Thanks to the co-first author [Yunfan Ye](https://github.com/yunfan1202).
## Citation
~~~
waiting for updating
~~~