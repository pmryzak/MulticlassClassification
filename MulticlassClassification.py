from keras.datasets import fashion_mnist
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
from keras.preprocessing import image
from keras.datasets import fashion_mnist

#loading data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#vectorizing images
images_train =  []
for image_train in x_train:
    images_train.append(image_train.flatten())
print(len(images_train))
images_test = []

for image_test in x_test:
    images_test.append(image_test.flatten())

images_train = np.array(images_train)
images_test = np.array(images_test)

#initiate classifier
fashion_mnist_classifier = OneVsRestClassifier(LogisticRegression(verbose=1, solver='lbfgs', max_iter=1000))
#fit classifier
fashion_mnist_classifier.fit(images_train, y_train);
#confusion matrix
conf_matrix = confusion_matrix(y_test, fashion_mnist_classifier.predict(images_test))
print("Confusion_matrix:")
print(conf_matrix)
#heatmap
sns.heatmap(conf_matrix)
#score
print(fashion_mnist_classifier.score(images_test, y_test))
plt.show()

pickle.dump(fashion_mnist_classifier, open('fashion_mnist_classifier.model', 'wb'))

#Single logistic classifier by learning probability distribution across multiple classes

fashion_mnist_classifier2 = LogisticRegression(verbose=1, max_iter=1000, multi_class="multinomial", solver="sag")

fashion_mnist_classifier2.fit(images_train, y_train)

conf_matrix2 = confusion_matrix(y_test, fashion_mnist_classifier2.predict(images_test))

print("Confusion_matrix2:")
print(conf_matrix2)
sns.heatmap(conf_matrix2)
fashion_mnist_classifier2.score(images_test, y_test)
plt.show()

print('Score #2 %s' % fashion_mnist_classifier2.score(images_test, y_test))

pickle.dump(fashion_mnist_classifier2, open('fashion_mnist_classifier2.model', 'wb'))
"""
fashion_mnist_classifier2__from_file = pickle.load(open('fashion_mnist_classifier2.model', 'rb'))

conf_matrix3 = confusion_matrix(y_test, fashion_mnist_classifier2__from_file.predict(images_test))
print("Confusion_matrix:")
print(conf_matrix)
sns.heatmap(conf_matrix)

print("Classifier2 from file ------------------------------------------")
print('Score %s' % fashion_mnist_classifier2__from_file.score(images_test, y_test))
plt.show()

img = image.load_img('pobrane.jpg').convert('LA')
img.save('greyscale1.png')

image_file = 'greyscale1.png'
img = image.load_img(image_file, target_size=(28, 28), grayscale=True, color_mode="grayscale")
x = image.img_to_array(img)

y = x.flatten().reshape(1, -1)
print("s")
print(fashion_mnist_classifier__from_file.predict(y))
"""