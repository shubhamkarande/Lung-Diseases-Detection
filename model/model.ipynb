{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "8f1ff135",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Admin\\anaconda3\\lib\\site-packages\\scipy\\__init__.py:146: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.23.0\n",
      "  warnings.warn(f\"A NumPy version >={np_minversion} and <{np_maxversion}\"\n"
     ]
    }
   ],
   "source": [
    "\n",
    "import numpy as np # linear algebra\n",
    "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
    "import matplotlib.pyplot as plt\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.optimizers import Adam\n",
    "from tensorflow.keras.utils import plot_model\n",
    "from tensorflow.keras.models import Model, Sequential\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau\n",
    "from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D\n",
    "from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "2d2ab5a9",
   "metadata": {},
   "outputs": [],
   "source": [
    "#pip install efficientnet"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "ee006a57",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "import tensorflow as tf\n",
    "import os\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "ac1ccc81",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "660da85e",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_dir='C:/Users/Admin/Desktop/3project/lung diseases/A Contemporary Technique for Lung Disease Prediction using Deep Learning/model/train'\n",
    "test_dir='C:/Users/Admin/Desktop/3project/lung diseases/A Contemporary Technique for Lung Disease Prediction using Deep Learning/model/test'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "614d67c4",
   "metadata": {},
   "outputs": [],
   "source": [
    "batch_size = 1\n",
    "epochs = 15\n",
    "img_height = 180\n",
    "img_width = 180"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "75af0e1b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 771 images belonging to 4 classes.\n"
     ]
    }
   ],
   "source": [
    "train_image_generator = ImageDataGenerator(rescale=1./255)  \n",
    "train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,directory=train_dir,shuffle=True,target_size=(img_height, img_width),class_mode='categorical')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "7e13816a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 304 images belonging to 4 classes.\n"
     ]
    }
   ],
   "source": [
    "val_image_generator = ImageDataGenerator(rescale=1./255)  \n",
    "val_data_gen = val_image_generator .flow_from_directory(batch_size=batch_size,directory=test_dir,shuffle=True,target_size=(img_height, img_width),class_mode='categorical')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "d831542b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.applications.vgg16 import VGG16\n",
    "#from tensorflow.keras.applications import EfficientNetB0\n",
    "from tensorflow.keras.applications import ResNet50, VGG19, MobileNetV2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "426ec18e",
   "metadata": {},
   "outputs": [],
   "source": [
    " base_model = VGG16(weights = 'imagenet', include_top=False, input_shape=(img_width, img_height, 3))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "467f9364",
   "metadata": {},
   "outputs": [],
   "source": [
    "base_model.trainable = False"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "c408ca39",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"vgg16\"\n",
      "_________________________________________________________________\n",
      " Layer (type)                Output Shape              Param #   \n",
      "=================================================================\n",
      " input_1 (InputLayer)        [(None, 180, 180, 3)]     0         \n",
      "                                                                 \n",
      " block1_conv1 (Conv2D)       (None, 180, 180, 64)      1792      \n",
      "                                                                 \n",
      " block1_conv2 (Conv2D)       (None, 180, 180, 64)      36928     \n",
      "                                                                 \n",
      " block1_pool (MaxPooling2D)  (None, 90, 90, 64)        0         \n",
      "                                                                 \n",
      " block2_conv1 (Conv2D)       (None, 90, 90, 128)       73856     \n",
      "                                                                 \n",
      " block2_conv2 (Conv2D)       (None, 90, 90, 128)       147584    \n",
      "                                                                 \n",
      " block2_pool (MaxPooling2D)  (None, 45, 45, 128)       0         \n",
      "                                                                 \n",
      " block3_conv1 (Conv2D)       (None, 45, 45, 256)       295168    \n",
      "                                                                 \n",
      " block3_conv2 (Conv2D)       (None, 45, 45, 256)       590080    \n",
      "                                                                 \n",
      " block3_conv3 (Conv2D)       (None, 45, 45, 256)       590080    \n",
      "                                                                 \n",
      " block3_pool (MaxPooling2D)  (None, 22, 22, 256)       0         \n",
      "                                                                 \n",
      " block4_conv1 (Conv2D)       (None, 22, 22, 512)       1180160   \n",
      "                                                                 \n",
      " block4_conv2 (Conv2D)       (None, 22, 22, 512)       2359808   \n",
      "                                                                 \n",
      " block4_conv3 (Conv2D)       (None, 22, 22, 512)       2359808   \n",
      "                                                                 \n",
      " block4_pool (MaxPooling2D)  (None, 11, 11, 512)       0         \n",
      "                                                                 \n",
      " block5_conv1 (Conv2D)       (None, 11, 11, 512)       2359808   \n",
      "                                                                 \n",
      " block5_conv2 (Conv2D)       (None, 11, 11, 512)       2359808   \n",
      "                                                                 \n",
      " block5_conv3 (Conv2D)       (None, 11, 11, 512)       2359808   \n",
      "                                                                 \n",
      " block5_pool (MaxPooling2D)  (None, 5, 5, 512)         0         \n",
      "                                                                 \n",
      "=================================================================\n",
      "Total params: 14,714,688\n",
      "Trainable params: 0\n",
      "Non-trainable params: 14,714,688\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "base_model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "061b07ec",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential\"\n",
      "_________________________________________________________________\n",
      " Layer (type)                Output Shape              Param #   \n",
      "=================================================================\n",
      " vgg16 (Functional)          (None, 5, 5, 512)         14714688  \n",
      "                                                                 \n",
      " flatten (Flatten)           (None, 12800)             0         \n",
      "                                                                 \n",
      " dense (Dense)               (None, 4)                 51204     \n",
      "                                                                 \n",
      "=================================================================\n",
      "Total params: 14,765,892\n",
      "Trainable params: 51,204\n",
      "Non-trainable params: 14,714,688\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "classifier=tf.keras.models.Sequential()\n",
    "classifier.add(base_model)\n",
    "classifier.add(Flatten())\n",
    "classifier.add(Dense(4,activation='softmax'))\n",
    "classifier.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "4e1afd38",
   "metadata": {},
   "outputs": [],
   "source": [
    "classifier.compile(optimizer='adam',\n",
    "              loss='categorical_crossentropy',\n",
    "              metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "17ce58b9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/3\n",
      "771/771 [==============================] - 117s 149ms/step - loss: 0.6669 - accuracy: 0.8016 - val_loss: 0.1437 - val_accuracy: 0.9441\n",
      "Epoch 2/3\n",
      "771/771 [==============================] - 102s 132ms/step - loss: 0.2556 - accuracy: 0.9118 - val_loss: 1.1609 - val_accuracy: 0.7961\n",
      "Epoch 3/3\n",
      "771/771 [==============================] - 121s 157ms/step - loss: 0.2051 - accuracy: 0.9300 - val_loss: 0.2401 - val_accuracy: 0.9145\n"
     ]
    }
   ],
   "source": [
    "history = classifier.fit(train_data_gen, epochs=3,\n",
    "validation_data= val_data_gen,)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "01d63efc",
   "metadata": {},
   "outputs": [],
   "source": [
    "classifier.save('lung.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "493a7369",
   "metadata": {},
   "outputs": [],
   "source": [
    "#pip install Pillow"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "3d34d3e5",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "ef5957ef",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "304/304 - 30s - 30s/epoch - 97ms/step\n"
     ]
    }
   ],
   "source": [
    "y=np.concatenate([val_data_gen.next()[1] for i in range(val_data_gen.__len__())])\n",
    "true_labels=np.argmax(y, axis=-1)\n",
    "prediction= classifier.predict(val_data_gen, verbose=2)\n",
    "prediction=np.argmax(prediction, axis=-1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "37c236dc",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "21ec7742",
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_confusion_matrix(cm, classes,\n",
    "                          normalize=False,\n",
    "                          title='Confusion matrix',\n",
    "                          cmap=plt.cm.Blues):\n",
    "    \"\"\"\n",
    "    This function prints and plots the confusion matrix.\n",
    "    Normalization can be applied by setting `normalize=True`.\n",
    "    \"\"\"\n",
    "    plt.imshow(cm, interpolation='nearest', cmap=cmap)\n",
    "    plt.title(title)\n",
    "    plt.colorbar()\n",
    "    tick_marks = np.arange(len(classes))\n",
    "    plt.xticks(tick_marks, classes, rotation=85)\n",
    "    plt.yticks(tick_marks, classes)\n",
    "\n",
    "    if normalize:\n",
    "        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]\n",
    "        print(\"Normalized confusion matrix\")\n",
    "    else:\n",
    "        print('Confusion matrix, without normalization')\n",
    "\n",
    "    print(cm)\n",
    "\n",
    "    thresh = cm.max() / 2.\n",
    "    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):\n",
    "        plt.text(j, i, cm[i, j],\n",
    "            horizontalalignment=\"center\",\n",
    "            color=\"white\" if cm[i, j] > thresh else \"black\")\n",
    "\n",
    "    plt.tight_layout()\n",
    "    plt.ylabel('True label')\n",
    "    plt.xlabel('Predicted label')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "b77a6579",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "import itertools\n",
    "import matplotlib.pyplot as plt\n",
    "cm = confusion_matrix(y_true=true_labels, y_pred=prediction)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "ed4e4bc5",
   "metadata": {},
   "outputs": [],
   "source": [
    "cm_plot_labels = ['COVID19','NORMAL','PNEUMONIA','TURBERCULOSIS']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "243d50d8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion matrix, without normalization\n",
      "[[96  0  0  0]\n",
      " [ 0 84 12  0]\n",
      " [ 0  1 63  0]\n",
      " [13  0  0 35]]\n",
      "Accuracy: 0.914\n",
      "Precision: 0.919\n",
      "Recall: 0.914\n",
      "F-Measure: 0.914\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')\n",
    "from sklearn.metrics import accuracy_score\n",
    "acc=accuracy_score(true_labels,prediction) \n",
    "print('Accuracy: %.3f' % acc)\n",
    "from sklearn.metrics import precision_score\n",
    "precision = precision_score(true_labels,prediction,labels=[1,2], average='micro')\n",
    "print('Precision: %.3f' % precision)\n",
    "from sklearn.metrics import recall_score\n",
    "recall = recall_score(true_labels,prediction, average='micro')\n",
    "print('Recall: %.3f' % recall)\n",
    "from sklearn.metrics import f1_score\n",
    "score = f1_score(true_labels,prediction, average='micro')\n",
    "print('F-Measure: %.3f' % score)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "7311748f",
   "metadata": {},
   "outputs": [],
   "source": [
    "from PIL import Image\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "def test_on_img(img):\n",
    "    data=[]\n",
    "    image = Image.open(img)\n",
    "    image = image.resize((224,224))\n",
    "    data.append(np.array(image))\n",
    "    X_test=np.array(data)\n",
    "    Y_pred = classifier.predict_classes(X_test)\n",
    "    return image,Y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9a0869a2",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
