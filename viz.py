#!/usr/bin/python
from __future__ import print_function
import numpy as np
import nmf
import data_yale
import matplotlib.pyplot as plt
import sys
import random


class ProgressViz:
    def __init__(self, n_iterations, nmf_type, W, n_rows, n_cols):
        plt.ion()
        self.n_rows, self.n_cols = n_rows, n_cols
        self.n_comp = W.shape[1]
        self.sub_rows, self.sub_columns = self.determine_subplots()
        self.figure, self.axes = plt.subplots(self.sub_rows, self.sub_columns)
        self.figure.suptitle(u"Loss and components -- NMF w/ {0}".format(nmf_type), size=10)
        self.ax_loss = self.axes[0, 0]
        self.ax_loss.set_title(u"Loss", size=8)
        self.lines, = self.ax_loss.plot([], [], u'o')
        self.images = []
        for i in xrange(self.sub_rows * self.sub_columns - 1):
            sub_i, sub_j = (1 + i) % self.sub_rows, (1 + i) / self.sub_rows
            subplot = self.axes[sub_i, sub_j]
            if i < self.n_comp:
                self.images.append(subplot.imshow(self.prepare_image(W[:, i]), cmap=u"Greys"))
                subplot.set_title(u"W[:, %d]" % i, size=8)
                subplot.set_axis_off()
            else:
                # Disable empty subplots
                subplot.set_visible(False)
        self.ax_loss.set_autoscaley_on(True)
        self.ax_loss.set_xlim(0, n_iterations)
        self.ax_loss.grid()
        self.ax_loss.get_xaxis().set_visible(False)
        self.ax_loss.get_yaxis().set_visible(False)

    def determine_subplots(self):
        nb_plots = self.n_comp + 1
        squared_root = np.sqrt(nb_plots)
        int_squared_root = int(squared_root)
        if int_squared_root == squared_root:
            return int_squared_root, int_squared_root
        else:
            return int_squared_root, int_squared_root + 1

    def update_draw(self, iterations, losses, W):
        # Update loss
        self.lines.set_xdata(iterations)
        self.lines.set_ydata(losses)
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()
        # Update mat' fact
        for i in xrange(self.n_comp):
            self.images[i].set_data(self.prepare_image(W[:, i]))
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def prepare_image(self, vec):
        return 1. - vec.reshape((self.n_rows, self.n_cols))

    def wait_end(self):
        plt.ioff()
        plt.show()


def main(folder=u"CroppedYale", nmf_type=u"divergence", r=10, n_iterations=10):
    cropped_yale = data_yale.load_cropped_yale(folder)
    n_rows, n_cols = cropped_yale[0].shape
    random.shuffle(cropped_yale)
    V = np.vstack((x.flatten() for x in cropped_yale[:500])).transpose() / 255.
    n, m = V.shape
    model = nmf.get_model(nmf_type, n, m, r)
    p_viz = ProgressViz(n_iterations, nmf_type, model.W, n_rows, n_cols)
    iterations, losses = [], []
    for i in xrange(n_iterations):
        model.update_factors(V)
        loss = model.compute_loss(V)
        print(u"epoch", i, u"loss", loss)
        losses.append(loss)
        iterations.append(i + 1)
        p_viz.update_draw(iterations, losses, model.W)
    print(u"Final:", model.compute_loss(V))
    p_viz.wait_end()


if __name__ == u"__main__":
    try:
        main(unicode(sys.argv[1]), unicode(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    except:
        main()
