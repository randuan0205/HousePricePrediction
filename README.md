# HousePricePrediction
A kaggle competition project

# Background:
This is a competition on Kaggle for house price prediction (these properties are in Ames, Iowa). Participants were given around 1,460 training sample with 80 features (both numerical and categorical) of houses such as size, number of rooms, and location, and condition. Even though the data size is small (only 1,400 samples for training), it is still an interesting and challenging project since there are a big amount of NA values and diversified plus correlated features. These challenges can help sharpen important ML skills such as data preprocessing and feature engineering. Plus, due to small data size, it is convenient to test more complex models such as Deep Learning (MLP – Multi Layer Perceptron is used .
# Goals: There are several goals I want to achieve through this task
1.	Test various traditional ML techniques on this dataset (data preprocessing, feature engineering, parameter tuning, XGBoost regressor)
2.	Compare deep learning with XGboost for model performance
3.	With generative AI becoming very hot these days, people are talking about the possibility that machine will take over software engineering. To test this idea for machine learning,  I used Cursor, a AI based coding tool, to build the machine learning model automatically. Then I compare its performance to my ‘man-made’ models.

# Results and Learning:
Let’s use data to tell the story, below is the performance metrics comparison among 3 models I build. 
## The first one is the validset’s RMSE (I separated 200 training samples as validset for the purpose of parameter tuning)
### Manually tuned XGBoost regressor: 0.1270, DNN (Multi-Layer Perceptron): 0.1253, XGBoost regressor auto-built by AI tool: 0.1561 
## The second one is the inference results returned by Kaggle:
### Manually tuned XGBoost regressor: 0.1221, DNN (Multi-Layer Perceptron): 0.1281, XGBoost regressor auto-built by AI tool: NA (did bother to try it due to the poor performance on validset)

# Based on these comparison results, here is the key insights I would like to share:
1. Gradient Boosting Tree is always a good choice for tabular data. Compared to Deep Neural Netowrk, it is easier for parameter tuning and save a lot of hardware investment (doesn't need GPUs)
2. Even though DNN is powerful for unstructured data, it is hard for it to beat GBT on tabular data. However, it also delivered very good result. If data sample has bigger size, DNN will be worth trying for better performance. However, DNN's higher demand for hardware expecially GPUS should be another factor to be considered.
3. Speaking of AI generated ML models, it seems lacking the ML domain knowledge of dealing with complex features, so the model didn’t perform well (got worse RMSE metric for valid set). However, I am pretty impressed with its coding capabilities. I feel it could be a coding assistant to AI engineers to improve efficiency. This direction definitely needs more testing in future.

# On the other hand: some insights obtained from manually tuning the model: 
1. It is always good to check features carefully and develop careful plan of feature cleaning and feature engineering.
2. Plus, some tradinal ML techniques (automatic feature selection (RFE used here), mean encoding for categorical features, parameter tuning through sklearn.RandomizedsearchCV) are very effective to help XGBoost Regressor learn the pattern from training data, finally delivering impressive performance.
3. I also tried to create new features through combining existing features and create more powerful ones. But this doesn't really help XGBoost regressor to learn (please refer the section 'Feature Engineering' in notebook). The learning here is that, for XGBoost, you'd better let it handle feature engineering by itself instead of manually creating new ones. XGBoost naturally can handle feature selection and feature interaction by using its algorthm.

