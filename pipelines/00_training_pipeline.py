from luigi.contrib.external_program import ExternalProgramTask
from luigi.parameter import IntParameter, Parameter
from luigi import LocalTarget, Task
from helper.keras_util import build_generator
from helper.cv2_util import calc_baseline_acc
from helper.model_util import define_model
import json
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import builder


"""
We want to download the dataset using "curl".
Luigi provides a baseclass named **ExternalProgramTask** to utilize external programs. 
It simply calls the external program with the provided commandline arguments. The output 
target can be referenced through *self.output()*.
"""
class DownloadDataset(ExternalProgramTask):

    dataset_version = IntParameter(default=1)
    dataset_name = Parameter(default="dataset")

    base_url = "http://plainpixels.work/resources/datasets"
    file_fomat = "zip"

    def output(self):
        return LocalTarget("/tmp/%s_v%d.%s" % (self.dataset_name,
                                               self.dataset_version,
                                               self.file_fomat))

    def program_args(self):
        url = "%s/%s_v%d.%s" % (self.base_url, 
                                self.dataset_name, 
                                self.dataset_version,
                                self.file_fomat)
        return ["curl", "-L",
                "-o", self.output().path,
                url]


"""
Just as before, we use **ExternalProgramTask** to unzip the archive. The major difference is 
that **ExtractDataset** now implements *requires(...)* and links to **DownloadDataset** as a
dependency. The required target can be referenced through *self.input()*.

*Input*: DownloadDataset <br>
*Output*: A folder containing the images
"""
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


"""
The configuration for the deep-learning model is essentially the Keras ImageDataGenerator. For 
the sake of simplicity we do not parameterize this task. But we can grasp the idea how to do it.

*Input*: Nothing required <br>
*Output*: A pickled ImageDataGenerator
"""
class Configure(Task):
    
    config_name = Parameter(default="standard")

    def output(self):
        return LocalTarget("configurations/%s.pickle" % self.config_name)

    def run(self):
        import pickle
        from tensorflow import keras
        self.output().makedirs()
        generator = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
        with self.output().open("w") as f:
            pickle.dump(generator, f)


"""
This task runs the baseline validation and saves it to a file. The same as before, flexibility can be greatly 
enhanced by als versioning the baseline validation.

*Input*: ExtractDataset, Configure <br>
*Output*: A JSON-File containing the baseline accuracy
"""
class BaselineValidation(Task):
    
    dataset_version = IntParameter(default=1)
    dataset_name = Parameter(default="dataset")
    config_name = Parameter(default="standard")

    validation_set = "Test"
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


"""
Task No.5 trains a Keras model and persists it to the filesystem.

*Input*: ExtractDataset, Configure <br>
*Output*: A .h5 file representing the model architecture and its weights
"""
class TrainModel(Task):
    
    dataset_version = IntParameter(default=1)
    dataset_name = Parameter(default="dataset")
    config_name = Parameter(default="standard")
    model_version = IntParameter(default=1)
    model_name = Parameter(default="keras_model")
    
    training_set = "Training"
    epochs = 8

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
        model = define_model(input_shape, num_classes)
        steps_per_epoch = training_data.samples // training_data.batch_size
        model.fit_generator(training_data,
                            steps_per_epoch=steps_per_epoch,
                            epochs=self.epochs,
                            verbose=2)
        model.save(self.output().path)


"""
The last task evaluates our model and - if it surpasses the baseline accuracy - saves the evaluation 
results to the filesystem. Let the task crash if the model does not perform well enough. It's worth 
an exception!

*Input*: ExtractDataset, Configure, TrainModel, BaselineValidation<br>
*Output*: A JSON file containing the evaluation results
"""
class Evaluate(Task):
    
    dataset_version = IntParameter(default=1)
    dataset_name = Parameter(default="dataset")
    config_name = Parameter(default="standard")
    model_version = IntParameter(default=1)
    model_name = Parameter(default="keras_model")

    validation_set = "Test"

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
        from tensorflow import keras
        self.output().makedirs()
        model_path = self.input()[0].path
        model = keras.models.load_model(model_path)
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


"""
The Keras model is performing well. Let's deploy it to TensorFlow Serving.

It can be loaded with TensorFlow Serving by the following command:
tensorflow_model_server --model_name="keras_model" --model_base_path="serving/keras_model"

Input: TrainModel, Evaluate
Output: The TensorFlow-Graph and its weights
"""
class Export(Task):
    dataset_version = IntParameter(default=1)
    dataset_name = Parameter(default="dataset")
    config_name = Parameter(default="standard")
    model_version = IntParameter(default=1)
    model_name = Parameter(default="keras_model")

    def requires(self):
        yield Evaluate(self.dataset_version,
                       self.dataset_name,
                       self.config_name,
                       self.model_version,
                       self.model_name)
        yield TrainModel(self.dataset_version,
                         self.dataset_name,
                         self.config_name,
                         self.model_version,
                         self.model_name)

    def output(self):
        return LocalTarget("serving/%s/%d" % (self.model_name,
                                              self.model_version))

    def run(self):
        from tensorflow import keras
        self.output().makedirs()
        model_path = self.input()[1].path
        model = keras.models.load_model(model_path)
        tensor_info_input = tf.saved_model.utils.build_tensor_info(model.input)
        tensor_info_output = tf.saved_model.utils.build_tensor_info(model.output)
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'input': tensor_info_input},
                outputs={'prediction': tensor_info_output},
                method_name=signature_constants.PREDICT_METHOD_NAME))

        export_path = self.output().path
        tf_builder = builder.SavedModelBuilder(export_path)
        with tf.keras.backend.get_session() as sess:
            tf_builder.add_meta_graph_and_variables(
                sess=sess,
                tags=[tag_constants.SERVING],
                signature_def_map={
                    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
                }
            )
            tf_builder.save()
