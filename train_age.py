import tensorflow
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, initializers, metrics
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

from time import time
import os
import glob
import pandas
import argparse

parser = argparse.ArgumentParser(description="Create age model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--image_size", type=int, default=128, help="Input image size")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--epoch", type=int, default=8, help="Epoch count")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
args = parser.parse_args()

size = args.image_size
batch_size = args.batch_size
epoch = args.epoch
lr = args.lr

checkpoint_path = "training_age/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
modelFileName = "model_age"
ageRange = 5
ageRangeHalf = ageRange/2

def age_mae(y_true, y_pred):
    true_age = K.sum(y_true * K.arange(ageRangeHalf, classesCount*ageRange + ageRangeHalf, ageRange, dtype="float32"), axis=-1)
    pred_age = K.sum(y_pred * K.arange(ageRangeHalf, classesCount*ageRange + ageRangeHalf, ageRange, dtype="float32"), axis=-1)
    return K.abs(true_age - pred_age)

folders = ["UTKFace/"]
data, labels = [], []
countCat, class_weight = {}, {}
cats = set()
meanAge = 0
for folder in folders:
    for file in glob.glob(folder + "*.jpg"):
        file = file.replace(folder, "")
        age, gender, race = file.split("_")[0:3]
        age, gender = int(age), int(gender)
        if age < 0 or age > 99:
            #print("Age error: " file)
            continue
        age = age//ageRange * 5
        catName = str(age).zfill(3)
        meanAge+=age
        if catName not in countCat:
            countCat[catName]=0
        countCat[catName]+=1
        data.append(folder + file)
        cats.add(catName)
        labels.append(catName)  

print("meanAge: " + str(meanAge/len(data)))
classesCount = len(countCat)
print("cat number: ", classesCount)
print("countCat:", countCat)
minVal = min(countCat.values())
print("minVal: ", minVal)
for key in countCat:
    class_weight[key] = 1
    class_weight[key]/=countCat[key]
    class_weight[key]*=minVal
print("class_weight:", class_weight)
print("categories:", cats)
train_df = pandas.DataFrame(data={"filename": data, "class": labels})

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filename",
        y_col="class",
        shuffle=True,
        target_size=(size, size),
        batch_size=batch_size,
        subset='training',
        class_mode='categorical')

print("class_indices:", train_generator.class_indices)
class_weight_tmp = {}
for key in train_generator.class_indices:
    class_weight_tmp[train_generator.class_indices[key]] = class_weight[key]

class_weight = class_weight_tmp
print("class_weight:", class_weight)

validation_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filename",
        y_col="class",
        target_size=(size, size),
        batch_size=batch_size,
        subset='validation',
        class_mode='categorical')

latest = tensorflow.train.latest_checkpoint(checkpoint_dir)
if not latest and os.path.exists(modelFileName):
    print("Load: " + modelFileName)
    classifier = load_model(modelFileName, custom_objects={"age_mae":age_mae})
else:
    classifier = tensorflow.keras.applications.mobilenet_v2.MobileNetV2(weights=None, input_tensor=None, input_shape=(size, size, 3), pooling="avg", classes=classesCount)
    if latest:
        print("Load: " + latest)
        classifier.load_weights(latest)

classifier.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=[age_mae])
#classifier.summary()

#python3 -m tensorboard.main --logdir logs/1
callbacks = [
    ModelCheckpoint(checkpoint_path, monitor='val_loss', mode='min', save_best_only=True, verbose=1, save_weights_only=True),
    TensorBoard(log_dir='logs/{}'.format(time()))]

history = classifier.fit(
        train_generator,
        shuffle=True,
        epochs=epoch,
        validation_data=validation_generator,
        class_weight=class_weight,
        callbacks=callbacks)

classifier.save(modelFileName)
