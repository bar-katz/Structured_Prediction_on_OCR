# OCR dataset classification using Structured Prediction approach

This project was made in order to practice Structured Prediction.

``structured_model.py`` uses structured prediction approach it takes in consideration the possibility of a letter being predicted given the last predicted letter(in the word). 

``simple_multiclass_model.py`` uses multiclass `W·x` in order to predict.

``simple_structured_model.py`` was an attempt to use structured prediction approach using `W·ϕ(x, y_hat)` in order to predict.

## Usage

1. Download Stanfords [OCR dataset](http://ai.stanford.edu/~btaskar/ocr/) and place the files in `data/`

2. Run
    ```sh
    python structured_model.py
    ```

3. (Optional) Run the 2 other models for comparison

## Results

| | simple multiclass model | simple structured model | structured model |
|:---:|:---:|:---:|:---:|
| Accuracy      | 75.218%      | 74.344% | 80.353% |

See what the structured model learned of the possibilities of bigram<br>
x axis - previous letter<br>
y axis - current letter<br>

For example look at the pairs (q,u), (l,y), (n,g): all of these get high value since they are likely
to appear together.<br>
On the other hand (u,u), (e,i), (a,a) will get low value.<br>

![bigarm_heatmap](https://user-images.githubusercontent.com/33622626/51331407-5bd0ec00-1a82-11e9-94b4-2f52d42ff7a5.png)


## Build With
* [seaborn](https://seaborn.pydata.org/) – data visualization package


## Author

Bar Katz – [bar-katz on github](https://github.com/bar-katz) – barkatz138@gmail.com
