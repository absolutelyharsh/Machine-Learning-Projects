# Mobile Price Classification

## Problem Statement:
Bob has started his own mobile company. He wants to give tough fight to big companies like Apple,Samsung etc.

He does not know how to estimate price of mobiles his company creates. In this competitive mobile phone market you cannot simply assume things. To solve this problem he collects sales data of mobile phones of various companies.

Bob wants to find out some relation between features of a mobile phone(eg:- RAM,Internal Memory etc) and its selling price. But he is not so good at Machine Learning. So he needs your help to solve this problem.

In this problem you do not have to predict actual price but a price range indicating how high the price is

## Dataset

Column Name and Descriptions

    battery_power = Battery Power in hz
    blue = Bluetooth available or not
    clock_speed = Microprocesssor Speed
    dual_sim = Has Dual Sim Card or Not
    fc = Front Camera Mega Pixels
    four_g = 4G or Not
    int_memory = Internal Memory in GB
    m_dep = Mobile Depth
    mobile_wt = Weight of mobile phone
    n_cores = Number of cores of processor
    pc = Primary Camera mega pixels
    px_height = Pixel Resolution Height
    px_width = Pixel Resolution Width
    ram = Random Access Memory in Megabytes
    sc_height = Screen Height of mobile in cm
    sc_width = Screen Width of mobile in cm
    talk_time = longest time that a single battery charge will last when you are
    three_g = Has 3G or not
    touch_screen = Has touch screen or not
    wifi = Has wifi or not
    price_range = 0 : low cost, 1: mid cost, 2: High Cost

Number of Train Data is 2000.<br>
Number of Test Data is 1000.<br>

There are no null values in train or test dataset.

## Models and Results

1. Logistic Regression without standardizing or scaling : 64.25%
2. KNN (10 Neighbors): 90.75%
3. Logistic Regression with Standardizing : 95.55%
4. Logistic Regression Best Features : 97.25%

