import numpy as np
import matplotlib.pyplot as plt

#First Step of the Project
# A function to plot images
def show_image(img):
    image = img.reshape((28, 28))
    plt.imshow(image, 'gray')


# Reading The Train Set
train_images_file = open('train-images.idx3-ubyte', 'rb')
train_images_file.seek(4)
num_of_train_images = int.from_bytes(train_images_file.read(4), 'big')
train_images_file.seek(16)

train_labels_file = open('train-labels.idx1-ubyte', 'rb')
train_labels_file.seek(8)

train_set = []
for n in range(num_of_train_images):
    image = np.zeros((784, 1))
    for i in range(784):
        image[i, 0] = int.from_bytes(train_images_file.read(1), 'big') / 256
    
    label_value = int.from_bytes(train_labels_file.read(1), 'big')
    label = np.zeros((10, 1))
    label[label_value, 0] = 1
    
    train_set.append((image, label))


# Reading The Test Set
# test_images_file = open('t10k-images.idx3-ubyte', 'rb')
# test_images_file.seek(4)
#
# test_labels_file = open('t10k-labels.idx1-ubyte', 'rb')
# test_labels_file.seek(8)
#
# num_of_test_images = int.from_bytes(test_images_file.read(4), 'big')
# test_images_file.seek(16)
#
# test_set = []
# for n in range(num_of_test_images):
#     image = np.zeros((784, 1))
#     for i in range(784):
#         image[i] = int.from_bytes(test_images_file.read(1), 'big') / 256
#
#     label_value = int.from_bytes(test_labels_file.read(1), 'big')
#     label = np.zeros((10, 1))
#     label[label_value, 0] = 1
#
#     test_set.append((image, label))


# Plotting an image
# show_image(train_set[0][0])
# print(train_set[0][1])
# plt.show()
#
# show_image(test_set[0][0])
# print(test_set[0][1])
# plt.show()

#Second Step of the Project

#Activation Function = Sigmoid
def sig(x):
 return 1/(1 + np.exp(-x))

#Weights with normal random numbers and zero biases
first_layer_weight = np.matrix(np.random.randn(16,784))
first_layer_bias = np.zeros(16).reshape(16,1)
second_layer_weight = np.matrix(np.random.randn(16,16))
second_layer_bias = np.zeros(16).reshape(16,1)
third_layer_weight = np.matrix(np.random.randn(10,16))
third_layer_bias = np.zeros(10).reshape(10,1)

#output of the network for 100 inputs
correct_output = 0
wrong_output = 0
#Calculate output of the network for 100 inputs and check the accuracy
for i in range(100):
    nn_inputs = train_set[i][0]
    first_Layer_output = sig(first_layer_weight * nn_inputs + first_layer_bias)
    second_layer_output = sig(second_layer_weight * first_Layer_output + second_layer_bias)
    nn_outputs = sig(third_layer_weight * second_layer_output + third_layer_bias)
    max_output = np.where(nn_outputs == np.amax(nn_outputs))[0][0]
    if max_output == np.where(train_set[i][1] == 1)[0][0]:
        correct_output += 1
    else:
        wrong_output += 1

print(correct_output/100)


