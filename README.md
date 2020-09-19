# LayoutNet in TensorFlow 2.3  

Reimplement of [LayoutNet](https://xtqiao.com/projects/content_aware_layout/) in TensorFLow 2.3.  

**Reference**:  
Zheng, Xinru, et al. "Content-aware generative modeling of graphic design layouts." ACM Transactions on Graphics (TOG) 38.4 (2019): 1-15.

## Introduction  

LayoutNet is a content-aware graphic design layout generation model. It is able to synthesize layout designs based on the visual and textual semantics of user inputs.  

## Example Results  

## LayoutNet on Colab  

We have put demo and the whole training pipeline on Colab, you can click [here](https://colab.research.google.com/drive/1IUkqqyevxW-N8hnXMIpx8JcAotkcTX4O?usp=sharing) to play with it.  

## Getting Started with Docker (Recommended when GPU ready, but Linux only)  

> With Docker, only Nvidia driver is required, no need to configure CUDA and cuDNN.  

## Getting Started on Local Machine  

> If you want to use GPU on local machine, make sure you have installed right version of CUDA and cuDNN before running the following command. You can check it from [here](https://www.tensorflow.org/install/source#build_the_package).  

```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Play with Pre-trained Model  

> - When using Docker, run the container in interactive mode.  
>
> ```shell
> docker run 
> ```
>
> - When running on local machine, make sure you have activated virtual environment.  
>
> ```shell
> source venv/bin/activate
> ```

Now you can run build-in test function to generate some samples of the LayoutNet.  

```shell
python main.py --test
```

## Train Model  

```shell
python main.py --train
```
