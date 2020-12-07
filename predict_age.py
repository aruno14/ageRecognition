import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy

image_size = (48, 48)
model_name = "model_age"


def age_mae(y_true, y_pred):
    classesCount=20
    true_age = K.sum(y_true * K.arange(2.5, classesCount*5+5, 5, dtype="float32"), axis=-1)
    pred_age = K.sum(y_pred * K.arange(2.5, classesCount*5+5, 5, dtype="float32"), axis=-1)
    return K.mean(K.abs(true_age - pred_age))

model = load_model(model_name, custom_objects={'age_mae':age_mae})
for imageName in ["test.jpg"]:
    print(imageName)
    test_image = image.load_img(imageName, target_size=image_size)
    test_image = image.img_to_array(test_image)
    test_image = test_image.astype('float32')
    test_image /= 255.0
    test_image = numpy.expand_dims(test_image, axis=0)
    prediction = model.predict(test_image)[0]
    print(prediction)
    age = numpy.sum(prediction * numpy.arange(2.5, len(prediction)*5, 5, dtype="float32"))
    print(age)
