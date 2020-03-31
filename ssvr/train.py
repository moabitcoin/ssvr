from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

from ssvr.dataset import RotatedDataset, random_split


def main(args):
    dataset = RotatedDataset(args.dataset, target_size=(224, 224),
                             preprocessing_function=preprocess_input,
                             data_format="channels_first", shuffle=True,
                             batch_size=1)

    num_train = int(len(dataset) * 0.9)
    num_val = len(dataset) - num_train

    trainset, valset = random_split(dataset, [num_train, num_val])

    model = ResNet50(classes=len(dataset.angles), weights=None)

    metrics = []
    metrics += [SparseCategoricalAccuracy()]

    # TODO: learning rate finder + one cycle policy

    optimizer = Adam(learning_rate=args.lr)
    loss = SparseCategoricalCrossentropy()

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    callbacks = []
    callbacks += [TensorBoard(histogram_freq=1)]
    callbacks += [ModelCheckpoint(filepath="model.hdf5", save_best_only=True)]
    callbacks += [EarlyStopping(patience=3)]

    history = model.fit_generator(generator=trainset, epochs=args.epochs,
                                  callbacks=callbacks, validation_data=valset,
                                  workers=1, use_multiprocessing=True,
                                  shuffle=True)

