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



<p align = "center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/RSggurOx6Ug" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>

<br>

---
### 🚗Training AWS Car 🚗
The goal was to create a custom reward function so that the AWS Deep Racer completes an unseen track as fastest as possible, and as accurately as possible. An example of a reward function would be:


```python

def reward_function(params):

  import math

def reward_function(params):

  # Read input parameters
  track_width = params['track_width']
  distance_from_center = params['distance_from_center']
  
  # reward function as Gauss curve with the variable distance_from_center
  reward = (1 / (math.sqrt(2 * math.pi * (track_width*2/15) ** 2)) * math.exp(-((distance_from_center + track_width/10) ** 2 / (4 * track_width*2/15) ** 2))) *(track_width*2/3)
  
  return float(reward)
    # - - - - -
    
    return speed_reward + heading_reward + steering_reward

```
 

<p align = "center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/At4k1VLL1bM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>
<br>

---
<center>© 2020 Khanh Tran. Powered by Jekyll and the Minimal Theme.</center>


