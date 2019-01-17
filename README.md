# OCR dataset classification using Structured Prediction approach

This project was made in order to practice Structured Prediction.

``structured_model.py`` uses structured prediction approach it takes in consideration the possibility of a letter being predicted given the last predicted letter(in the word). 

``simple_multiclass_model.py`` uses multiclass `W·x` in order to predict.

``simple_structured_model.py`` was an attempt to use structured prediction approach using `W·ϕ(x, y_hat)` in order to predict.

## Usage

1. Download Stanfords [OCR dataset](http://ai.stanford.edu/~btaskar/ocr/)

2. Run
    ```sh
    python structured_model.py
    ```

3. (Optional) Run the 2 other models for comparison

## Results

| | simple multiclass model | simple structured model | structured model |
|:---:|:---:|:---:|:---:|
| Accuracy      | 75.218%      | 74.344% | 80.353% |


## Build With
* [seaborn](https://seaborn.pydata.org/) – data visualization package


## Author

Bar Katz – [bar-katz on github](https://github.com/bar-katz) – barkatz138@gmail.com