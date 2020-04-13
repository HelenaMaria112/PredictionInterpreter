import matplotlib.pyplot as plt
import datetime
import os.path
import os
import errno

def save(name, plt = None, fig = None):
    desktop = os.path.normpath(os.path.expanduser("~/Desktop"))
    plotPath = desktop + "/PredictionInterpreter Plots/"
    try:
        os.makedirs(plotPath)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    graphName = plotPath + name + str(datetime.datetime.now())[:19].replace(":","-") + ".png"
    if plt is not None:
        plt.savefig(graphName, dpi=500, bbox_inches = "tight")
        print(graphName + " plotted")
        plt.show()
    if fig is not None:
        fig.savefig(graphName)
        print(graphName + " plotted")
        fig.show()
