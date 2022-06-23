
# Discovering Object that Can Move

This is the repository for [*Discovering Object that Can Move*](https://arxiv.org/abs/2203.10159), published at CVPR 2022.  


[[Project Page](https://zpbao.github.io/projects/CVPR22-Discovering/)]
[[Paper](https://zpbao.github.io/projects/CVPR22-Discovering/)]

<img src='./imgs/pipeline.png' />

## Currently the training code is 

## Set up

Python 3 dependencies:

* torch 1.7.1+CUDA11.0 
* matplotlib
* cv2
* numpy
* scipy
* tqmd

The default training scipt takes ~15GB*4 for training


## Running code
### pre-processing
To run the model on the TRI-PD dataset, first download PD data and annotations and unzip the tar files. Then run the commend

```python merge_pd.py --pd_dir /A/B --additional_dir /C/D``` 

The full dataset will be merged to the path ```/A/B```

### Train the model 

``` python trian.py``` 

See ```run.sh``` for a sample training script

### Evaluating the pre-trained model

To evaluate or infer on the test set, first download the pre-trained model (or train it with the training code), then run

```python eval.py``` 

to compute the FG.ARI score. Notice that evaluating on the whole test set is time-consumpting... in ```eval.py``` we randomly sample five frames for each video among the whole set and evaluate FG.ARI score on that sub video. The ARI score may be slightly different compared with evaluating on the whole set. To evaluate on the whole set, change the random index in ```dataset.py``` to a fixed number and evaluate every 5 frames. 

To  infer on a video of arbitary length, run

```python infer.py```

See ```eval.sh``` and ```infer.sh``` for a sample evaluating script.



## TRI-PD Dataset and pre-trained models
pre-trained models updating soon...

PD datasets:   
[[Raw dataset]](https://tri-ml-public.s3.amazonaws.com/datasets/tri-pd-flow/tri_pd_flow_00.tar): The full PD dataset contains RGB, semantic segmentation, instance segmentation, optical flow, depth, camera colibrations, 2D/3D bounding boxes

Additional annotations and pre-trained model:   
[[Additional data]](https://cmu.box.com/s/2cf718pyuh9jhg7davq23y79kdinq4aq)



## Citation

```
@inproceedings{bao2022discovering,
    Author = {Zhipeng Bao, Pavel Tokmakov, Allan Jabri, Yu-Xiong Wang, Adrien Gaidon, and Martial Hebert},
    Title = {Discorying Object that Can Move},
    Booktitle = {CVPR},
    Year = {2022},
}
```


