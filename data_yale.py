from __future__ import print_function
import numpy as np
import re
import glob
import os


def load_pgm(filename, byteorder=u">"):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, u"rb") as f:
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
    except:
        print(u"ignoring image in", filename)
        return None

def load_cropped_yale(folder):
    print(u"loading images in", folder)
    loaded = [load_pgm(f) for f in glob.glob(os.path.join(folder, u"*.pgm"))]
    return [x for x in loaded if np.any(x, None)]

