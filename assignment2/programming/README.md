# HW2 Programming

|   ID    |    Name     |
| :-----: | :---------: |
| 1953902 | GAO Yangfan |

[TOC]

## 1. RANSAC implementation

### a. idea & result

There are 5 steps of RANSAC:

1. Randomly select samples to fit a model
2. Compute the number of 'inliers' (oppose to outliers) using certain metric of distance
3. If the number of 'inliers' is more than a threshold $T$ terminate the program
4. Else if the model is better than the previous one (more 'inliers'), modify the model
5. Repeat 2-4 steps for $K$ times

$T$ can be set manually but $K$ can be estimated automatically according to [this](https://zhuanlan.zhihu.com/p/62238520). Here are the steps of estimating $K$

1. Manually set a probability of getting a good model $P$
2. Assume the ratio of 'inliers' is $r$
3. For each estimate, the probability of picking at least 1 outlier is $1-r^2$
4. So we have $P = 1-(1-r^2)^K$
5. Then $K = \dfrac{\log(1-P)}{\log(1-r^2)}$

However, in this problem, as the amount of data is small, the estimation of $K$ may be unused in the program.

The result is as shown below:

1.  using Manhattan distance:

![image-20220430205638095](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220430205638095.png)

2.  using distance to the model(line)

![image-20220430205712086](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220430205712086.png)

### b. how to run this

1. `cd RANSAC `
2. `pip install -r requirements.txt`
3. `python main.py`

## 2. BEV generator

According to the requirements, I will only explain the files I submitted to you.

1. `BEV-submit/intrinsics_of_my_camera.txt` contains the intrinsic parameters of the camera of my mobile phone.
2. `BEV-submit/img.jpg` is the original image
3. `BEV-submit/result.jpg` is generated the bird-eye view image 
