
from pathlib import Path
import cv2
import matplotlib
from io import BytesIO
import numpy as np
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils.lib import log

_SUPPORTED_IMG_TYPES = [".jpg", ".png", ".JPEG", ".PNG", ".jpeg"]


def load_image(filename, grayscale=True):

    filename = Path(filename)
    assert Path(filename).exists(), f"File {filename} does not exist"
    extension = filename.suffix
    assert (
        extension in _SUPPORTED_IMG_TYPES
    ), f"File extension {extension} not supported"

    image = cv2.imread(str(filename))

    if grayscale and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def matplotlib_figure_to_numpy(fig):

    try:
        if matplotlib.get_backend() == "Qt5Agg":
            canvas = FigureCanvasQTAgg(fig)
        elif matplotlib.get_backend() == "TkAgg":
            canvas = FigureCanvasTkAgg(fig)
        elif matplotlib.get_backend() == "agg":
            canvas = FigureCanvasAgg(fig)
        else:
            buf = BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            image = Image.open(buf).convert("RGB")
            image = np.array(image)[..., :3]
            buf.close()
            return image

        canvas.draw()

        if matplotlib.get_backend() == "Qt5Agg":
            image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        else:
            image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)

        width, height = fig.canvas.get_width_height()
        image = image.reshape((height, width, 3))
        return image
    except:
        log.warning("Could not convert figure to numpy array.")
        return np.array([])
