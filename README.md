# CNN-based Gomoku AI

# File Directory
```bash
├── main.py
├── Menu.py
├── Rule.py
├── utils.py
├── Opago
│   ├── dataset
│   │   ├── 00000.npz
│   │   ├── ***
│   │   └── 03983.npz
│   ├── model
│   │  └── model_27.pth
│   ├── pre_dataset
│   │   ├── Renju1
│   │   ├── Renju2
│   │   └── Renju3
│   ├── cnn_utils.py
│   ├── create_dataset.py
│   ├── dataloader.py
│   ├── SimpleNet.py
│   └── train.py
├── image
│   ├── board.jpg
│   ├── black.png
│   └── white.png
├── making_board.ipynb
``` 

# Prerequisites
- **`python 3.9`**
- **`pytorch 1.12`**


# Project Description
- This project constructs an Opago (오목 모델, Gomoku AI model) AI with the pygame module in Python   
- A convolutional neural network (CNN) is used
- The model structure is very simple and the hyperparameters are already tuned 


## Dataset
- The dataset is from https://gomocup.org/results/
- Input: an image from during a play
- Output: an image that instructs the next step of the input image
- We can get a lot of data from just one game
- There is tremendous data available on that aforementioned website, which is organized by year
- This project itself uses game data from the years 2021, 2020 and 2019

## Files Description
### pre_dataset
+ Coordinate information from the website above
+ Renju is one of the names that refer to a way of playing Gomoku with specific rules (which are also the most popular)
### create_dataset.py & dataset
+ Change coordinate information to numpy 15 * 15 arrays and save them as `.npz` files in the dataset folder in order to be used as input to the CNN network
### dataloader.py & cnn_utils.py & SimpleNet.py & train.py
- Train Opago AI model using the CNN method.
- The input size is `(batchsize, 1, 15, 15)`, the output size is `(batchsize, 255)` 
- To obtain a probabilty estimation, we use the sigmoid function as the activation function in the last layer with a 255 size array
- We can augment the data by eight times by flipping (up, down,left, right) and rotating (90, 180, 270, 360 degrees) since the board is symmetric
### Menu.py
- Checking  whether the total game is over or not by counting the number of wins whether it exceeds two times
- Also, showing message if the game is over
### Rule.py
- Checking  whether the game is over or not with an algorithm that is checking that there are 5 stones in arrow
### utils.py
- There are various functions necessary to visualize the game from drawing the stone on a board to letting the AI get a point to win a game
### main.py 
- If the whole game is over, then quit the game
### making_board.ipynb
- Making a board image with a cv2 and PIL module

## Game Specification
1. Use a 15*15 grid board
2. Randomly assign the black and white player
3. Black always begins
4. The black player's first stone must be placed in the center of the board
5. If any player wins twice, the whole game is over
6. Blocking samsam which is the strategy that black cannot use is not implemented yet
7. The second black stone cannot be placed inside 5x5 center area (since Black should be penalized for being able to place the first stone)

