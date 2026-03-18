import torch
import numpy as np
import random
import torch.nn as nn
import pandas as pd
import time


torch.set_num_threads(8)
torch.set_num_interop_threads(1)

seed = 128
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

torch.use_deterministic_algorithms(True)




df = pd.read_csv("datasets/mnist_train.csv", header = None)


x = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32)
y = torch.tensor(df.iloc[:, 0].values, dtype=torch.long)
x = x / 255.0

train_size = int(0.8 * len(x))

train_x = x[:train_size]
train_y = y[:train_size]

test_x = x[train_size:]
test_y = y[train_size:]



def create_model():
    
    model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    )

    return model



def benchmark(batch_size, epochs):

    model = create_model()

    sgd = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    start = time.time()

    for epoch in range(epochs):

        g = torch.Generator()
        g.manual_seed(seed + epoch)
        perm = torch.randperm(len(train_x), generator=g)

        shuffled_x = train_x[perm]
        shuffled_y = train_y[perm]


        for i in range(0, len(train_x), batch_size):
            batch_x = shuffled_x[i : i + batch_size]
            batch_y = shuffled_y[i : i + batch_size]

            sgd.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()
            sgd.step()

    end = time.time()
    print(f"[batch_size = {batch_size}, epochs = {epochs} => time :  {(end - start):.2f}s, ", end="")

    with torch.no_grad():
        output = model(test_x)
        predicted = torch.argmax(output, dim = 1)
        accuracy = (predicted == test_y).float().mean()
        print(f"accuracy : {(accuracy * 100 ):.3f}]")





def benchmark_adam(batch_size, epochs):

    model = create_model()
    adam = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(epochs):

        g = torch.Generator()
        g.manual_seed(seed + epoch)
        perm = torch.randperm(len(train_x), generator=g)

        shuffled_x = train_x[perm]
        shuffled_y = train_y[perm]

        for i in range(0, len(train_x), batch_size):
            batch_x = shuffled_x[i : i + batch_size]
            batch_y = shuffled_y[i : i + batch_size]

            adam.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()
            adam.step()

    end = time.time()
    print(f"[batch_size = {batch_size}, epochs = {epochs} => time :  {(end - start):.2f}s, ", end="")

    with torch.no_grad():
        output = model(test_x)
        predicted = torch.argmax(output, dim = 1)
        accuracy = (predicted == test_y).float().mean()
        print(f"accuracy : {(accuracy * 100 ):.3f}]")





if __name__ == "__main__":


    print(f"====[Benchmark for MNIST dataset with 60k samples]====")
    print(f"  --> Neuralnetwork: 784 x 128 x 128 x10")
    print(f"  --> Activation functions : ReLU ReLU Softmax")
    print(f"  --> Loss function : Cross Entropy")
    print(f"  -->learnrate : 0.001 (for both runs)")


    print(f"====[Mini Batch Gradient descent:]=====================")
    benchmark(batch_size=1, epochs=1)
    benchmark(batch_size=2, epochs=1)
    benchmark(batch_size=4, epochs=1)
    benchmark(batch_size=8, epochs=1)
    benchmark(batch_size=16, epochs=20)
    benchmark(batch_size=32, epochs=20)
    benchmark(batch_size=64, epochs=20)


    print(f"====[Mini batch gradient descent with Adam and L2 regulazation enabled]====")
    print(f" --> beta1 = 0.9, beta2 = 0.999, epsilon = 10e-8, lambda = 10e-4")
    benchmark_adam(batch_size=1, epochs=1)
    benchmark_adam(batch_size=2, epochs=1)
    benchmark_adam(batch_size=4, epochs=1)
    benchmark_adam(batch_size=8, epochs=1)
    benchmark_adam(batch_size=16, epochs=20)
    benchmark_adam(batch_size=32, epochs=20)
    benchmark_adam(batch_size=64, epochs=20)

    











