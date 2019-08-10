# Spark twitter sentiment analysis
This Repository implements a machine learning model which analyzes tweets and predicts if they are positive, or negative. The programming laguage is scala. 
## Setup

## Dataset Description
In this repository Twitter [dataset from Kaggle](https://www.kaggle.com/c/twitter-sentiment-analysis2/data) is used. The training set contains 100k examples, test set has 300k examples. The data is provided in CSV format. Data is very irregular and requires preprocessing
```
+ - - - - - - - - - - - - - - - - - - - - - - - - - - - +
| ItemID | Sentiment |       SentimentText              |
+ - - - -+ - - - - - + - - - - - - - - - - - - - - - - -+
|   1    |     0     | is so sad for my APL friend..... |
|   2    |     0     | I missed the New Moon trailer... |
|   3    |     1     |    omg its already 7:30 :O       |
|  ...   |    ...    |    .......................       |
+ - - - - - - - - - - - - - - - - - - - - - - - - - - - +
```
## Training process structure
The flow of the whole model development is outlined by the shceme below 
<p>
<img src="https://i.imgur.com/yBkrWse.png" alt="drawing" width="550"/>
</p>

## ML Model 
