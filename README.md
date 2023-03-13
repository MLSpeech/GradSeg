# GradSeg: Unsupervised Word Segmentation Using Temporal Gradient Pseudo-Labels (ICASSP 2023)


Tzeviya Sylvia Fuchs (fuchstz@cs.biu.ac.il) \
Yedid Hoshen (yedid.hoshen@mail.huji.ac.il)             
 

GradSeg is an unsupervised approach for word segmentation using pretrained deep self-supervised features. It uses the temporal gradient magnitude of the embeddings (the distance between the embeddings of subsequent frames) to define psuedo-labels for word centers, and trains a linear classifier using these psuedo-lables. It then uses the classifier score to predict whether a frame is a word or a boundary.


If you find our work useful, please cite: 
```
@inproceedings{fuchs23_icassp,
  author={Tzeviya Sylvia Fuchs and Yedid Hoshen},
  title={{Unsupervised Word Segmentation Using Temporal Gradient Pseudo-Labels}},
  year=2023,
  booktitle={ICASSP 2023},
}
```


------


## Installation instructions

- Python 3.8+ 

- Pytorch 1.10.0

- torchaudio 0.10.0

- numpy

- boltons

- Download the code:
    ```
    git clone https://github.com/MLSpeech/GradSeg.git
    ```


## How to use

In this example, we will demonstrate how to run GradSeg on the [Buckeye](https://buckeyecorpus.osu.edu/) corpus. 

- We use the same experimental setup as in "DSegKNN: Unsupervised Word Segmentation using K Nearest Neighbors (INTERSPEECH 2022)" ([Paper](https://arxiv.org/pdf/2204.13094.pdf), [Code](https://github.com/MLSpeech/DSegKNN), see README file there for data preprocessing).


- Run ```grad_segmenter.py``` with the following options:


	```
	python grad_segmenter.py --min_separation 3 
				 --train_n 100 
				 --eval_n -1 
				 --reg 1e7 
				 --target_perc 20 
				 --frames_per_word 15
				 --train_path datasets/buckeye_split/train/
				 --val_path datasets/buckeye_split/val/

	```

	Result should be:

	```
	Final result: 45.77675489067894 44.84274602637809 45.30493707647628 -2.040356216886474 53.62285114327586
	```

	which are the `precision`, `recall`, `F-score`, `OS`, and `R-value`.


- For comparison, the evaluation script ```eval_segmentation.py``` used here is by [Herman Kamper](https://github.com/kamperh/vqwordseg/blob/main/eval_segmentation.py).

