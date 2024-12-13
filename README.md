# RNNs as a Generative Model

This work generates the bottom half of MNIST digits using a Recurrent Neural Network (RNN). The task involves training an RNN to predict the bottom half of the image given the top half. The model learns to generate patches sequentially and reconstructs the full image.

The goal is to train an RNN that can predict the bottom half of a 28x28 MNIST digit image given its top half. The process involves:

1. Dividing each 28x28 image into **16 non-overlapping patches** of size 7x7.
2. Feeding the **first 8 patches** (top half) into the RNN.
3. Predicting the **next 8 patches** (bottom half) sequentially using the trained RNN.
4. Combining the top and bottom halves to reconstruct the full image.
   
### Training
The RNN is trained to predict the next patch in a sequence, as described by:


$(Y_t, C_{t+1}, H_{t+1}) = \text{RNN}(X_t, C_t, H_t)$

Where:
- $Y_t$: Prediction for patch $\( t+1 \)$
- $X_t$: Input patch at time $\( t \)$
- $C_t, H_t$: Memory cell and hidden state at time $t$ 
- $C_0, H_0$: Initialized to zeros

The loss function compares each prediction $Y_t$ with the ground truth $X_{t+1}$:

$L = \sum_{t=2}^{16} \sum_{d=1}^{49} D(X_{t,d} || Y_{t-1,d})$

Where:
- $D(\cdot || \cdot)$: Distance metric (Mean Squared Error is used)

### Generation
To generate the bottom half of an image:
1. Feed the **first 8 patches** $(X_1, X_2, ..., X_8)$ into the trained model.
2. Predict patch-by-patch using:


$(Y_9, C_{10}, H_{10}) = \text{RNN}(Y_8, C_9, H_9)$

$(Y_{10}, C_{11}, H_{11}) = \text{RNN}(Y_9, C_{10}, H_{10})$

...and so on until:

$(Y_{15}, C_{16}, H_{16}) = \text{RNN}(Y_{14}, C_{15}, H_{15})$

3. Combine the known top-half patches $(X_1, ..., X_8)$ with the predicted bottom-half patches $(Y_9, ..., Y_{15})$.

### Solution Overview

#### Data Preprocessing
1. **Patch Division**: Each image is divided into a sequence of **16 patches** of size $7 \times 7$, flattened into vectors of size 49.
2. **Training Data**: The first 15 patches are used as input $( X_1, ..., X_{15})$, and their corresponding next patches $( X_2, ..., X_{16})$ are used as targets.

#### Model Architecture
The model is an RNN implemented with LSTM or GRU layers:
- Input: Sequence of shape `(None, 49)` (patches)
- Layers:
    - Two stacked LSTM layers with hidden size of `64`
    - Dense layer with `49` units for predicting each patch
- Output: Sequence of predicted patches

#### Training
The model is trained using Mean Squared Error (MSE) loss and early stopping to prevent overfitting.

### Results

#### Generated Images
For each digit class (0–9), we generate **10 images** by feeding their top halves into the model and generating their bottom halves sequentially.



