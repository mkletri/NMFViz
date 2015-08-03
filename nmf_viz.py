#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import random
import logging
import simplejson as json
import re
import glob
import os
import gzip
import struct
import array


def load_pgm(filename, byteorder=">"):
    """Return image data from a raw PGM file as numpy array.
    Format specification: http://netpbm.sourceforge.net/doc/pgm.html
    """
    with open(filename, "rb") as f:
        buff = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buff).groups()
    except AttributeError:
        raise ValueError(u"Not a raw PGM file: '%s'" % filename)
    try:
        a = np.frombuffer(buff, dtype=u"u1" if int(maxval) < 256 else byteorder + u"u2", count=int(width) * int(height),
                          offset=len(header)).reshape((int(height), int(width)))
        return a
    except Exception as e:
        logging.warning("Ignoring image in %s for reason %s", filename, str(e))
        return None

def load_mnsit(filename):
    logging.info("Loading MNIST data from %s", filename)
    with gzip.open(filename) as gf:
        magic, size, rows, cols = struct.unpack(">IIII", gf.read(16))
        if magic != 2051:
            raise IOError("Magic number was expected to be <2049> but was <%d>" % magic)
        data = array.array("B", gf.read())
    data = [np.array(data[i * rows * cols : (i + 1) * rows * cols]) for i in range(size)]
    logging.info("Loaded %d images from %s", len(data), filename)
    return data, rows, cols


def load_cropped_yale(folder):
    paths = [_ for _ in glob.glob(os.path.join(folder, u"*.pgm"))]
    logging.info("Loading %d images in %s", len(paths), folder)
    loaded = [load_pgm(f) for f in glob.glob(os.path.join(folder, u"*.pgm"))]
    loaded = [x for x in loaded if np.any(x, None)]
    logging.info("Successfully loaded %d images out of %d", len(loaded), len(paths))
    n_rows, n_cols = loaded[0].shape
    logging.info("Images dimensions: %d by %d pixels", n_rows, n_cols)
    return loaded, n_rows, n_cols


def load_cbcl(filename):
    logging.info("Loading data from %s", filename)
    with open(filename, "r") as f:
        n_examples = int(f.readline())
        n_features = int(f.readline())
        assert n_features == 361, "Expected number of features to be <361> but was <%d>" % n_features
        data = [np.array([float(x) for x in line.strip().split()[:-1]]) for line in f]
        logging.info("Loaded %d images from %s", n_features, filename)
    return data, 19, 19


def load_data(conf):
    t = conf["type"]
    if t == "Cropped Yale":
        data, n_rows, n_cols = load_cropped_yale(conf["path"])
        logging.info("Shuffling images...")
        random.shuffle(data)
        n_images = min(conf["number"], len(data))
        logging.info("Converting to flat vectors, keeping %d images...", n_images)
        data = np.vstack((x.flatten() for x in data[:conf["number"]])).transpose() / 255.0
    elif t == "MNIST":
        data, n_rows, n_cols = load_mnsit(conf["path"])
        logging.info("Shuffling images...")
        random.shuffle(data)
        n_images = min(conf["number"], len(data))
        logging.info("Converting to a matrix")
        data = np.vstack((_ for _ in data[:conf["number"]])).transpose() / 255.0
    elif t == "CBCL":
        data, n_rows, n_cols = load_cbcl(conf["path"])
        logging.info("Shuffling images...")
        random.shuffle(data)
        n_images = min(conf["number"], len(data))
        logging.info("Converting to a matrix")
        data = np.vstack((_ for _ in data[:conf["number"]])).transpose()
    else:
        raise ValueError("Invalid type of data: %s (expecting 'Cropped Yale', 'MNIST' or 'CBCL')" % t)
    return data, n_rows, n_cols


class NonnegativeMatrixFactorization:
    """
    "Abstract" non-negative matrix factorization.
    """

    def __init__(self, n_features, n_examples, components, iterations, loss_name, random_seed=0):
        self.n_features = n_features
        self.n_examples = n_examples
        self.components = components
        self.iterations = iterations
        self.loss_name = loss_name
        np.random.seed(random_seed)
        self.W = np.random.random((n_features, components))
        self.H = np.random.random((components, n_examples))


class EuclideanLeeSeungNonnegativeMatrixFactorization(NonnegativeMatrixFactorization):
    """
    Implementation of the update rules for Mean Squared Error loss as in the paper from Lee & Seung:
    Algorithms for non-negative matrix factorization (NIPS 2001)
    """

    def __init__(self, n_features, n_examples, components, iterations):
        NonnegativeMatrixFactorization.__init__(self, n_features, n_examples, components, iterations, "euclidean")

    def update_factors(self, V):
        self.H *= np.dot(np.transpose(self.W), V) / np.dot(np.dot(np.transpose(self.W), self.W), self.H)
        self.W *= np.dot(V, np.transpose(self.H)) / np.dot(self.W, np.dot(self.H, np.transpose(self.H)))

    def compute_loss(self, V):
        return np.linalg.norm(V - np.dot(self.W, self.H)) ** 2 / self.n_examples


class DivergenceLeeSeungNonnegativeMatrixFactorization(NonnegativeMatrixFactorization):
    """
    Implementation of the update rules for divergence loss (linked to Kullback-Leibler divergence) as in the paper from
    Lee & Seung: Algorithms for non-negative matrix factorization (NIPS 2001)
    """

    def __init__(self, n_features, n_examples, components, iterations):
        NonnegativeMatrixFactorization.__init__(self, n_features, n_examples, components, iterations, "divergence")

    def update_factors(self, V):
        # The [:, None] is a trick to force correct broadcasting for np.divide
        self.H *= np.dot(np.transpose(self.W), V / np.dot(self.W, self.H)) / np.sum(self.W, axis=0)[:, None]
        self.W *= np.dot(V / np.dot(self.W, self.H), np.transpose(self.H)) / np.sum(self.H, axis=1)

    def compute_loss(self, V):
        # Compute WH only once.
        WH = np.dot(self.W, self.H)
        return np.sum(V * np.log(1e-10 + V / WH) - V + WH) / self.n_examples


class SparseHoyerNonnegativeMatrixFactorization(NonnegativeMatrixFactorization):
    """
    Implementation of a sparse nonnegative matrix factorization as in the paper from Patrik O. Hoyer:
    Non-negative sparse coding (arXiv)
    """
    def __init__(self, n_features, n_examples, components, iterations, sparseness, learning_rate, decay):
        NonnegativeMatrixFactorization.__init__(self, n_features, n_examples, components, iterations, "sparse")
        self.sparseness = sparseness
        self.learning_rate = learning_rate
        self.decay = decay
        self.W = np.where(self.W < 0.5, 0, self.W)
        self.H = np.where(self.H < 0.5, 0, self.H)

    def update_factors(self, V):
        self.H *= np.dot(np.transpose(self.W), V) / (np.dot(np.dot(np.transpose(self.W), self.W), self.H)
                                                    + self.sparseness)
        self.W += self.learning_rate * np.dot(V - np.dot(self.W, self.H), self.H.transpose())
        self.W = np.maximum(0, self.W)
        self.learning_rate *= self.decay

    def compute_loss(self, V):
        return np.linalg.norm(V - np.dot(self.W, self.H)) ** 2 / self.n_examples

class SparseL2NonnegativeMatrixFactorization(NonnegativeMatrixFactorization):
    """
    Own implementation: sparse on H and L2 on W.
    """
    def __init__(self, n_features, n_examples, components, iterations, sparseness, l2, learning_rate, decay):
        NonnegativeMatrixFactorization.__init__(self, n_features, n_examples, components, iterations, "sparse L2")
        self.sparseness = sparseness
        self.learning_rate = learning_rate
        self.decay = decay
        self.l2 = l2
        self.W = np.where(self.W < 0.5, 0, self.W)
        self.H = np.where(self.H < 0.5, 0, self.H)

    def update_factors(self, V):
        self.H *= np.dot(np.transpose(self.W), V) / (np.dot(np.dot(np.transpose(self.W), self.W), self.H)
                                                    + self.sparseness)
        self.W += self.learning_rate * (np.dot(V - np.dot(self.W, self.H), self.H.transpose()) - self.l2 * self.W)
        self.W = np.maximum(0, self.W)
        self.learning_rate *= self.decay

    def compute_loss(self, V):
        return np.linalg.norm(V - np.dot(self.W, self.H)) ** 2 / self.n_examples

def get_model(n_features, n_examples, conf):
    t = conf["type"]
    k = conf["components"]
    i = conf["iterations"]
    if t == "euclidean":
        logging.info("Creating nonnegative matrix factorization using Euclidean loss")
        return EuclideanLeeSeungNonnegativeMatrixFactorization(n_features, n_examples, k, i)
    elif t == "divergence":
        logging.info("Creating nonnegative matrix factorization using KL-Divergence loss")
        return DivergenceLeeSeungNonnegativeMatrixFactorization(n_features, n_examples,  k, i)
    elif t == "sparse":
        logging.info("Creating nonnegative matrix factorization using Hoyer's sparse loss")
        s = conf["sparseness"]
        l = conf["learning rate"],
        d = conf["learning rate decay"]
        return SparseHoyerNonnegativeMatrixFactorization(n_features, n_examples,  k, i, s, l, d)
    elif t == "sparse-l2":
        logging.info("Creating nonnegative matrix factorization using own sparse + L2 loss")
        s = conf["sparseness"]
        l = conf["learning rate"]
        d = conf["learning rate decay"]
        l2 = conf["l2"]
        return SparseL2NonnegativeMatrixFactorization(n_features, n_examples,  k, i, s, l2, l, d)
    else:
        raise ValueError("Invalid NMF type: {0}".format(conf["type"]))


class ProgressViz:
    def __init__(self, model, n_rows, n_cols):
        plt.ion()
        self.n_rows, self.n_cols = n_rows, n_cols
        self.n_comp = model.W.shape[1]
        self.sub_rows, self.sub_columns = self.determine_subplots()
        self.figure, self.axes = plt.subplots(self.sub_rows, self.sub_columns)
        self.figure.suptitle(u"Loss and components -- NMF w/ {0}".format(model.loss_name), size=10)
        self.ax_loss = self.axes[0, 0]
        self.ax_loss.set_title(u"Loss", size=8)
        self.lines, = self.ax_loss.plot([], [], u'o')
        self.images = []
        for i in range(self.sub_rows * self.sub_columns - 1):
            sub_i, sub_j = (1 + i) % self.sub_rows, (1 + i) / self.sub_rows
            subplot = self.axes[sub_i, sub_j]
            if i < self.n_comp:
                self.images.append(subplot.imshow(self.prepare_image(model.W[:, i]), cmap=u"Greys"))
                subplot.set_title(u"W[:, %d]" % i, size=8)
                subplot.set_axis_off()
            else:
                # Disable empty subplots
                subplot.set_visible(False)
        self.ax_loss.set_autoscaley_on(True)
        self.ax_loss.set_xlim(0, model.iterations)
        self.ax_loss.grid()
        self.ax_loss.get_xaxis().set_visible(False)
        self.ax_loss.get_yaxis().set_visible(False)

    def determine_subplots(self):
        nb_plots = self.n_comp + 1
        int_squared_root = int(np.sqrt(nb_plots))
        return int_squared_root, 1 + int(nb_plots / int_squared_root)

    def update_draw(self, iterations, losses, W):
        # Update loss
        self.lines.set_xdata(iterations)
        self.lines.set_ydata(losses)
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()
        # Update mat' fact
        for i in range(self.n_comp):
            self.images[i].set_data(self.prepare_image(W[:, i]))
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def prepare_image(self, vec):
        return 1. - vec.reshape((self.n_rows, self.n_cols))

    def wait_end(self):
        plt.ioff()
        plt.show()


def main(configuration):
    logging.info("Setting seed for random generator to %d", configuration["seed"])
    data_matrix, n_rows, n_cols = load_data(configuration["data"])
    random.seed(configuration["seed"])
    n_features, n_examples = data_matrix.shape
    logging.info("Data matrix dimensions: %d (features) by %d (examples)", n_features, n_examples)
    model = get_model(n_features, n_examples, configuration["nmf"])
    p_viz = ProgressViz(model, n_rows, n_cols)
    iterations, losses = [], []
    for i in range(model.iterations):
        model.update_factors(data_matrix)
        loss = model.compute_loss(data_matrix)
        logging.info("Iteration % 4d => loss: %f", i, loss)
        losses.append(loss)
        iterations.append(i + 1)
        p_viz.update_draw(iterations, losses, model.W)
    logging.info(u"Final loss: %f", model.compute_loss(data_matrix))
    p_viz.wait_end()


if __name__ == u"__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    with open("conf.json", "r") as cf:
        main(json.load(cf))
