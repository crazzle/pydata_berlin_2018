from luigi.contrib.external_program import ExternalProgramTask
from luigi.parameter import IntParameter, Parameter
from luigi import LocalTarget, Task
from helper.keras_util import build_generator
from helper.cv2_util import calc_baseline_acc
import json
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import uuid


class DownloadDataset(ExternalProgramTask):

    dataset_version = Parameter()
    dataset_name = Parameter()

    base_url = "http://plainpixels.work/resources/datasets"
    file_fomat = "zip"
    uid = uuid.uuid4()

    def output(self):
        return LocalTarget("/tmp/%s.%s" % (self.uid, self.file_fomat))

    def program_args(self):
        url = "%s/%s_v%d.%s" % (self.base_url, 
                                self.dataset_name, 
                                self.dataset_version,
                                self.file_fomat)
        return ["curl", "-L",
                "-o", self.output().path,
                url]


class ExtractDataset(ExternalProgramTask):
    
    dataset_version = IntParameter(default=1)
    dataset_name = Parameter(default="dataset")
    
    def requires(self):
        return DownloadDataset(self.dataset_version, self.dataset_name)

    def output(self):
        return LocalTarget("datasets/fruit-images-dataset/%d" % self.dataset_version)

    def program_args(self):
        self.output().makedirs()
        return ["unzip", "-u", "-q",
                "-d", self.output().path,
                self.input().path]


class Configure(Task):
    
    config_name = Parameter(default="standard")

    def output(self):
        return LocalTarget("configurations/%s.pickle" % self.config_name)

    def run(self):
        import pickle
        from keras.preprocessing.image import ImageDataGenerator 
        self.output().makedirs()
        generator = ImageDataGenerator(rescale=1. / 255)
        with self.output().open("w") as f:
            pickle.dump(generator, f)


class BaselineValidation(Task):
    
    dataset_version = IntParameter(default=1)
    dataset_name = Parameter(default="dataset")
    config_name = Parameter(default="standard")

    validation_set = "Test"
    img_height = 100
    img_width = 100
    baseline_name = "find_round_objects.json"

    def requires(self):
        yield ExtractDataset(self.dataset_version, self.dataset_name)
        yield Configure(self.config_name)

    def output(self):
        return LocalTarget("baseline/%s.json" % self.baseline_name)

    def run(self):
        dataset = self.input()[0].path
        config = self.input()[1].path
        test_data = build_generator(config, dataset, self.validation_set)
        result = calc_baseline_acc(test_data, dataset, self.validation_set)
        with self.output().open("w") as f:
            json.dump(result, f)


class TrainModel(Task):
    
    dataset_version = IntParameter(default=1)
    dataset_name = Parameter(default="dataset")
    config_name = Parameter(default="standard")
    model_version = IntParameter(default=1)
    model_name = Parameter(default="keras_model")
    
    training_set = "Training"
    img_height = 100
    img_width = 100
    epochs = 1

    def requires(self):
        yield ExtractDataset(self.dataset_version, self.dataset_name)
        yield Configure(self.config_name)

    def output(self):
        return LocalTarget("model/%d/%s.h5" % (self.model_version, self.model_name))

    def run(self):
        self.output().makedirs()
        dataset = self.input()[0].path
        config = self.input()[1].path
        training_data = build_generator(config, dataset, self.training_set)
        input_shape = training_data.image_shape
        num_classes = len(training_data.class_indices)
        model = self.define_model(input_shape, num_classes)
        steps_per_epoch = training_data.samples // training_data.batch_size
        model.fit_generator(training_data,
                            steps_per_epoch=steps_per_epoch,
                            epochs=self.epochs,
                            verbose=2)
        model.save(self.output().path)

    def define_model(self, input_shape, num_classes):
        model = Sequential()
        model.add(
            Conv2D(filters=4, kernel_size=(2, 2), strides=1, activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=8, kernel_size=(2, 2), strides=1, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Flatten())
        model.add(Dense(units=32, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(units=num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer="adam",
                      metrics=['accuracy'])
        return model


class Evaluate(Task):
    
    dataset_version = IntParameter(default=1)
    dataset_name = Parameter(default="dataset")
    config_name = Parameter(default="standard")
    model_version = IntParameter(default=1)
    model_name = Parameter(default="keras_model")

    validation_set = "Test"
    img_height = 100
    img_width = 100

    def requires(self):
        yield TrainModel(self.dataset_version, 
                         self.dataset_name, 
                         self.config_name,
                         self.model_version,
                         self.model_name)
        yield BaselineValidation(self.dataset_version,
                                 self.dataset_name,
                                 self.config_name)
        yield ExtractDataset(self.dataset_version, 
                             self.dataset_name)
        yield Configure(self.config_name)

    def output(self):
        return LocalTarget("evaluation/%d/%s.json" % (self.model_version, self.model_name))

    def run(self):
        self.output().makedirs()
        model_path = self.input()[0].path
        model = load_model(model_path)
        dataset = self.input()[2].path
        config = self.input()[3].path
        test_data = build_generator(config, dataset, self.validation_set)
        evaluation = model.evaluate_generator(test_data)

        with self.input()[1].open("r") as i:
            baseline_acc = json.load(i)["acc"]
        acc = evaluation[1]
        if acc > baseline_acc:
            result = {"acc": acc, "baseline_acc": baseline_acc}
            with self.output().open("w") as o:
                json.dump(result, o)
        else:
            raise Exception("Acc %f is smaller than baseline acc %f!" % (acc, baseline_acc))
