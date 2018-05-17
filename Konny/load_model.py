import keras
import numpy as np
from PIL import Image
from keras import backend


# im = Image.open('image.png')
# small = im.resize((28,28))



def predict(small):
    backend.clear_session()
    with open('model.json', 'r') as f:
        json = f.read()
    model = keras.models.model_from_json(json)

    # load weights into new model
    model.load_weights("model.h5")

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])
    Xbild_alt = np.array(small)
    Xbild = 255 - Xbild_alt
    Xbild = Xbild.reshape((1,784))
    y_pred = model.predict([Xbild])
    return np.argmax(y_pred)
