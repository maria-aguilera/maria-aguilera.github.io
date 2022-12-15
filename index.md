# Portfolio
## Machine Learning 
###  🚵‍♀️Kaggle Competition: Predict Forest Cover Type using Naive Bayes, KNN Classifier, XGBoost, Random Forest and Extra Trees🚵‍♀️ 


[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/forest-cover-type-classification.html)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/maria-aguilera/forest-cover-type-prediction/blob/main/forest-cover-type-classification.ipynb)
[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/15ZeArEeEWVRx-fLyK-ZtKPeqxLpz3H_B)

<div style="text-align: justify"> The task was to predict the forest cover type (the predominant kind of cover) from strictly catographic variables. Comprehensive exploratory data anlysis to understand the importance and significance of the variables variables, identifying  outliers, correlations and peforming feature engineering to increase the accuracy of the prediction..</div>

*  ***Skills***: Python | Matplotlib | Seaborn | Plotly | Scikit- learn Pipelines | Grid search | Hyperparameter Configuration | Data Visualization
<br>
<center><img src="images/forest-cover-type2.png"/></center>

---
### 🚀Kaggle Competition: Predicting whether passenger was sent to another dimension using manu models and parameters🚀

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/spaceship-titanic.html)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/maria-aguilera/spaceship-titanic)
[![Run Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1GgTDc8bqNdoxLfQ9bRVZQ1OFKEhcYz4x#scrollTo=kpC_nDA0OkkQ&uniqifier=1)
<br>
<div style="text-align: justify">Focus on data cleaning, feature relationship, missing values, data cleaning, feature engineering and modeling pipelines with very useful visuals.</div>

* The data can be downloaded in [Kaggle](https://www.kaggle.com/competitions/spaceship-titanic)
* ***Skills***: Data Cleaning | Pipeline | GridSearch | Missing Values | Data Exploration | Crossvalidation
<br>
<center><img src ="images/Age.png" width = "1000"/> </center>
<center><img src="images/Model Results.png" /></center>
<table><tr>
<td><img src="images/CabinRegion.png" style= "width: 500" /></td>
<td> <img src="images/Homeplanet and Deck Relationship.png" style= "width: 500" /> </td>
</tr></table>


---
### 🚲 Predict number bicycle users on an hourly basis🚲

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/bike-sharing.html)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/maria-aguilera/bike-sharing/blob/main/2_Group_A_Bike_Sharing.ipynb)
[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1ryydAJtLAJnjCMqVtIGNvLQh4qfEnSXc#scrollTo=TflFLV0YPFi1)

<div style="text-align: justify">Goal was to predict the total number of Washington D.C bicycle users on an hourly basis.</div>

* ***Skills***: Exploratory Data Analysis | Data Cleaning & Analysis | Time- Based Cross Validation | Python
<br>
<center><img src = "images\bike-sharing.png"/></center>
<center><img src="images\work_not_working.png" /></center>
<br>


---
## Social Network Analysis
### ☎ Analyzing Instagram Data Set using Graphs ☎

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/forest-cover-type-classification.html)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/maria-aguilera/graph-analysis/blob/main/sna-instagram-network-analysis.ipynb)
[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1ryydAJtLAJnjCMqVtIGNvLQh4qfEnSXc#scrollTo=TflFLV0YPFi1)

<div style="text-align: justify"> Using graph analysis, analyze the instagram data set available in Kaggle to find out the most influential members of the network to increase sales by advertisement. The dataset was too large to process, we therefore had to do exploratory data analysis to check how to reduce it so that it doesn't become a random network.</div>

* ***Skills***: GraphX | Comunity Detection Algorithims
<br>
<center><img src="images/gephi.jpeg" width = "500px"/></center>
<br>

---
## Reinforcement Learning
### 🌔 Lundar Landing Assignment 🌔

[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Open_in_Hugging_Face-blue)](https://huggingface.co/maria-aguilera/ppo-LunarLander-v2)
[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/19m9fhUUrC8mJQzE6Ilieipgss8JIw4Xq#scrollTo=V7JUZKw5kq0m)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/maria-aguilera/lunar-landing-ppo)

Our goal is to teach the Lunar Lander (our agent) how to correctly land their spaceship between two flags (our landing pad). The more accurately the agent is able to land, the bigger the ultimate reward he will be able to attain. The agent may choose any of the following four actions at any moment to achieve this objective: fire the left engine, fire the right engine, fire down the engine, or do nothing.


<center>
<video src= "images\rl\ll_Trim.mp4" controls="controls" style="max-width: 400px;"></center>
</video>
<br>

---
### 🚗Training AWS Car 🚗
The goal was to create a custom reward function so that the AWS Deep Racer completes an unseen track as fastest as possible, and as accurately as possible. An example of a reward function would be:


```python

def reward_function(params):

    # Reward weights
    speed_weight = 100
    heading_weight = 100
    steering_weight = 50

    # Initialize the reward based on current speed
    max_speed_reward = 10 * 10
    min_speed_reward = 3.33 * 3.33
    abs_speed_reward = params['speed'] * params['speed']
    speed_reward = (abs_speed_reward - min_speed_reward) / (max_speed_reward - min_speed_reward) * speed_weight
    
    # - - - - - 
    
    # Penalize if the car goes off track
    if not params['all_wheels_on_track']:
        return 1e-3
    
    # - - - - - 
    
    # Calculate the direction of the center line based on the closest waypoints
    next_point = params['waypoints'][params['closest_waypoints'][1]]
    prev_point = params['waypoints'][params['closest_waypoints'][0]]

    # Calculate the direction in radius, arctan2(dy, dx), the result is (-pi, pi) in radians
    track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0]) 
    # Convert to degree
    track_direction = math.degrees(track_direction)

    # Calculate the difference between the track direction and the heading direction of the car
    direction_diff = abs(track_direction - params['heading'])
    if direction_diff > 180:
        direction_diff = 360 - direction_diff
    
    abs_heading_reward = 1 - (direction_diff / 180.0)
    heading_reward = abs_heading_reward * heading_weight
    
    # - - - - -
    
    # Reward if steering angle is aligned with direction difference
    abs_steering_reward = 1 - (abs(params['steering_angle'] - direction_diff) / 180.0)
    steering_reward = abs_steering_reward * steering_weight

    # - - - - -
    
    return speed_reward + heading_reward + steering_reward

```

<center>
<video src= "images\rl\VID_20220607_160239_Trim.mp4"controls="controls" style="max-width: 500px;"></center>
</video>
<br>

---
<center>© 2020 Khanh Tran. Powered by Jekyll and the Minimal Theme.</center>


