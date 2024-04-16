# MNIST classification
### 1. Number of model parameters 
- computating the parameters of CustomMLP : {(32x32xa)+a} + {(axb)+b} + {(bx10)+10} ≒ 61,706 
<img width="600" alt="파라미터 수" src="https://github.com/jiwwnn/mnist_classification/assets/134251617/6a06f596-8040-470d-b0ed-5dd2d07d974b">

### 2. Plots of average loss and accuracy
- LeNet-5
<img src="https://github.com/jiwwnn/mnist_classification/assets/134251617/64fc400a-51b7-483f-bb13-29fbf2f3054f.png"  width="600">

- CustomMLP
<img src="https://github.com/jiwwnn/mnist_classification/assets/134251617/9a3696ff-30e2-408b-944c-410c0abd99b0.png"  width="600">

### 3. Comparing Performances of LeNet-5 and CustomMLP
- Result : LeNet-5 > CustomMLP 
  - Test Accuracy of LeNet-5 : 0.988 
  - Test Accuracy of CustomMLP : 0.970

### 4. Improving LeNet-5 using regularization techniques
- Methods 
  - Dropout 
  - Batch Normalization
- Result : got accuracy 0.990, which is higher than the original LeNet-5 (+0.02)
