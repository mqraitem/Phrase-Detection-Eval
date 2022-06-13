### Phrase Detection Evaluation
A python 3 code base to evaluate vision language models on [phrase detection](https://arxiv.org/abs/1811.07212).

Dependancies: `numpy`, `cython_bbox` 

You can run the evaluation on two datasets: Refcoco+ and Flickr30k Entities. For each dataset, follow the instructions: 

- Download the dataset images. [Flickr30k Entities](http://hockenmaier.cs.illinois.edu/DenotationGraph/)/[RefCOCO+](https://cocodataset.org/#download): You should download: train2014. 
- Inside the repo data/[DATASET] folder, there are im_ids.pkl and phrases.pkl. Read the image ids in order, then for each image: 
  - load the image 
  - pass the image along with phrases in phrases.pkl to your V+L model
  - output a 2d numpy array with dimensions: num_phrases X 5, where each line follows the structure: 
    - x1,y1,x2,y2,score (where the first 4 numbers are the box coordinates that best fit a given phrase) and score is a similarity score between the region and phrase. 
- Append each 2D numpy score array from each image to a list. 
- Save that list into a pickle file. 

You can check out a sample file [here](https://drive.google.com/drive/folders/1nPUe8VwP7eM5bl6bMjMYUiyhlWkEYSAy?usp=sharing). Then, run the following: 

`python python eval.py --dets_path [Path to your scores file from above] --dataset [DATASET]`

Where DATASET is either refcoco+ or flickr. 

NOTE: The files under data/DATASET were pickled using python 2.7. Therefore, make sure you have encoding='latin1' in your pickle.load. 
