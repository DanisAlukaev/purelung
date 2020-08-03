import tensorflow as tf
import os


class Segmentation:

    """Class used for context-aware image crop in the Chest X-ray images."""

    def __init__(self):
        """
        Load the trained model and create new tensorflow session.
        """
        self.graph = Segmentation.load_graph(os.path.join(os.path.split(__file__)[0], 'model_weights/retina_net.pb'))
        self.session = tf.compat.v1.Session(graph=self.graph)

    def predict(self, images):
        """
        Create binary masks of lungs, heart and clavicles for input set of images.

        :param images: a set of images to be segmented.
        :return: a list of binary masks for input set of images.
        """
        one_image = False
        if len(images.shape) == 3:
            one_image = True
            images.shape = (1, images.shape[0], images.shape[1], images.shape[2])
        result = self.session.run("sigmoid/Sigmoid:0", feed_dict={"data:0": images})
        result[result >= 0.5] = 1
        result[result < 0.5] = 0
        if one_image:
            return result[0]
        else:
            return result

    def predict_lungs(self, images):
        """
        Create binary masks of lungs for input set of images.

        :param images: a set of images to be segmented.
        :return: a list of binary masks for input images.
        """
        return self.predict(images)[..., 0]

    def predict_heart(self, images):
        """
        Create binary masks of heart for input set of images.

        :param images: a set of images to be segmented.
        :return: a list of binary masks for input images.
        """
        return self.predict(images)[..., 2]

    def predict_clavicles(self, images):
        """
        Create binary masks of clavicles for input set of images.

        :param images: a set of images to be segmented.
        :return: a list of binary masks for input images.
        """
        return self.predict(images)[..., 4]

    @staticmethod
    def load_graph(graph_path):
        """Load frozen TensorFlow graph."""
        with tf.io.gfile.GFile(graph_path, "rb") as graph_file:
            graph_definition = tf.compat.v1.GraphDef()
            graph_definition.ParseFromString(graph_file.read())
        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(graph_definition, name="")
        return graph
