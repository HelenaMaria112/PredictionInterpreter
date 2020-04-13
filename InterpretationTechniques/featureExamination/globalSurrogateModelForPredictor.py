from Connections.predictor import *
import matplotlib.pyplot as plt
import pandas as pd
import cython # only needed for smt. Needs to be installed extra.
from smt.utils import compute_rms_error
from smt.utils.options_dictionary import OptionsDictionary
from smt.problems import Sphere, NdimRobotArm, Rosenbrock
from smt.sampling_methods import LHS
from smt.surrogate_models import LS, QP, KPLS, KRG, KPLSK, GEKPLS
from smt.surrogate_models import IDW, RBF, RMTC, RMTB
from smt.problems import Sphere, NdimRobotArm, Rosenbrock
from smt.utils.options_dictionary import OptionsDictionary
from InterpretationTechniques.PlotAndShow import *

class Model():
    def __init__(self, fun, ttd, ndim= None, xL = None):
        if ndim is None and xL is None:
            self.t, self.title, self.xtest, self.ytest = fun(ttd.xt, ttd.yt, ttd.xtest, ttd.ytest)
        elif xL is None:
            self.t, self.title, self.xtest, self.ytest = fun(ttd.xt, ttd.yt, ttd.xtest, ttd.ytest, ndim)
        elif ndim is None:
            self.t, self.title, self.xtest, self.ytest = fun(ttd.xt, ttd.yt, ttd.xtest, ttd.ytest, xL)
        else:
            self.t, self.title, self.xtest, self.ytest = fun(ttd.xt, ttd.yt, ttd.xtest, ttd.ytest, xL, ndim)

def decode(encodingDictionaryGknto, predictions):
    return predictions.astype(float)
    return predictions[predictions.apply(lambda x :encodingDictionaryGknto[int(x)])]

def testTrainData(data, pr, encoded):
    if encoded:
        #data =  pr.encode(data, exceptedColumns=["prediction"] )
        data = pr.encode(data)
        data = data.astype(float)

    #drop columns with inf and nans:
    print(len(data))
    data.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    print(len(data))

    data = data.drop(pr.resultColumn, axis = 1)
    even_nos = [num for num in range(len(data)) if num % 2 == 0]
    uneven_nos = [num for num in range(len(data)) if num % 2 != 0]
    trainingData=data.iloc[even_nos].to_numpy()
    testData=data.iloc[uneven_nos].to_numpy()
    trainTestData = pd.Series()
    trainTestData["xt"]=trainingData[:,:-1]
    trainTestData["yt"]=trainingData[:,-1]
    trainTestData["xtest"]=testData[:,:-1]
    trainTestData["ytest"]=testData[:,-1]

    return trainTestData

def globalSurrogateModel(data, pr):
    pr.callingFunction =  "globalSurrogate"
    data["prediction"] = pr.predict(data)

    ttdE = testTrainData(data, pr, encoded=True)
    ttdNE = testTrainData(data, pr, encoded=False)

    options=OptionsDictionary()
    options.declare("ndim", 1.0, types=float)
    options.declare("return_complex", False, types=bool)
    xlimits = np.zeros((int(options["ndim"]), 4))
    xlimits[:, 0] = -2.0
    xlimits[:, 1] = 2.0
    funXLimits=xlimits
    ndim=4

    # models retrieved from https://github.com/SMTorg/smt/blob/master/tutorial/SMT_Tutorial.ipynb
    models = pd.Series()
    models["linear"] = Model(linearModel, ttdE)
    models["quadratic"] = Model(quadraticModel, ttdE)
    # models["kriging"] = Model(kriging, ttdE, ndim=ndim)
    # models["KPLSK"] = Model(kPLSK, ttdE)
    models["IDW"] = Model(iDW, ttdE)
    models["RBF"] = Model(rBF, ttdE)
    #models["RMTBSimba"] = Model(rMTBSimba, ttdNE,  xL = funXLimits)
    #models["GEKPLS"] = Model(gEKPLS, ttdNE, xL = funXLimits, ndim = ndim)
    #models["DEKPLS"] = Model(dEKPLS, ttdNE, xL = funXLimits, ndim = ndim)

    for model in models:
        compareToOrig(model.t, model.title, model.xtest, model.ytest)

def compareToOrig(t, title, xtest, ytest):

    y = t.predict_values(xtest)

    # Plot prediction/true values
    fig = plt.figure()
    plt.plot(ytest, ytest, '-', label='$y_{true}$')
    plt.plot(ytest, y, 'r.', label='$\hat{y}$')

    plt.xlabel('$y_{true}$')
    plt.ylabel('$\hat{y}$')

    plt.legend(loc='upper left')
    plt.title(title + ' validation of the prediction model')
    save(title, plt)

def linearModel(xt, yt, xtest, ytest):  #model needs floats
    # Initialization of the model
    t = LS(print_prediction=False)
    # Add the DOE
    t.set_training_values(xt, yt)

    # Train the model
    t.train()
    title='LS model: validation of the prediction model'
    # Prediction of the validation points
    print('RBF,  err: ' + str(compute_rms_error(t, xtest, ytest)))
    return t, title, xtest, ytest

def quadraticModel(xt, yt, xtest, ytest):
    ########### The QP model

    t = QP(print_prediction=False)
    t.set_training_values(xt, yt)

    t.train()
    title = 'QP model: validation of the prediction model'
    print('QP,  err: ' + str(compute_rms_error(t, xtest, ytest)))
    return t, title, xtest, ytest

def kriging(xt, yt, xtest, ytest, ndim):
    ########### The Kriging model

    # The variable 'theta0' is a list of length ndim.
    t = KRG(theta0=[1e-2] * ndim, print_prediction=False)
    t.set_training_values(xt, yt)
    t.train()

    title='Kriging model: validation of the prediction model'
    print(title)
    print('Kriging,  err: ' + str(compute_rms_error(t, xtest, ytest)))
    print("theta values", t.optimal_theta)

    # Plot the function, the prediction and the 95% confidence interval based on
    # the MSE
    return t, title, xtest, ytest

    # Value of theta

def kPLSK(xt, yt, xtest, ytest):
    ########### The KPLSK model
    # 'n_comp' and 'theta0' must be an integer in [1,ndim[ and a list of length n_comp, respectively.

    t = KPLSK(n_comp=2, theta0=[1e-2, 1e-2], print_prediction=False)
    t.set_training_values(xt, yt)
    t.train()

    print('KPLSK,  err: ' + str(compute_rms_error(t, xtest, ytest)))
    title= 'KPLSK model: validation of the prediction model'
    return t, title, xtest, ytest

def iDW(xt, yt, xtest, ytest):
     ########### The IDW model

    t = IDW(print_prediction=False)
    t.set_training_values(xt, yt)
    t.train()

    # Prediction of the validation points
    y = t.predict_values(xtest)
    print('IDW,  err: ' + str(compute_rms_error(t, xtest, ytest)))
    title='IDW'
    return t, title, xtest, ytest

def rBF(xt, yt, xtest, ytest):
    t = RBF(print_prediction=False, poly_degree=0)
    t.set_training_values(xt, yt)
    t.train()

    # Prediction of the validation points
    y = t.predict_values(xtest)
    print('RBF,  err: ' + str(compute_rms_error(t, xtest, ytest)))
    # Plot prediction/true values

    title='RBF model'
    return t, title, xtest, ytest

def rMTBSimba(xt, yt, xtest, ytest, funXLimits):
    t = RMTB(xlimits=funXLimits, min_energy=True, nonlinear_maxiter=20, print_prediction=False)
    t.set_training_values(xt, yt)
    # Add the gradient information
    #    for i in range(ndim):
    #        t.set_training_derivatives(xt,yt[:, 1+i].reshape((yt.shape[0],1)),i)
    t.train()

    # Prediction of the validation points
    print('RMTB,  err: ' + str(compute_rms_error(t, xtest, ytest)))
    # plot prediction/true values
    title='RMTB'
    return t, title, xtest, ytest

def rMTCSimba(xt, yt, xtest, ytest, funXLimits):
        t = RMTC(xlimits=funXLimits, min_energy=True, nonlinear_maxiter=20, print_prediction=False)
        t.set_training_values(xt, yt)
        t.train()

        # Prediction of the validation points
        print('RMTC,  err: ' + str(compute_rms_error(t, xtest, ytest)))
        title='RMTC model'
        return t, title, xtest, ytest

def dEKPLS(xt, yt, xtest, ytest, funXLimits, ndim):
    # 'n_comp' must be an integer in [1,ndim[,  'theta0' a list of n_comp values

    t = GEKPLS(n_comp=1, theta0=[1e-2], xlimits=funXLimits, delta_x=1e-2, extra_points=1, print_prediction=False)
    t.set_training_values(xt, yt)
    # Add the gradient information
    for i in range(ndim):
        t.set_training_derivatives(xt, yt[:, 1 + i].reshape((yt.shape[0], 1)), i)
    t.train()

    title='GEKPLS model'
    return t, title, xtest, ytest

def gEKPLS(xt, yt, xtest, ytest, funXLimits, ndim):
    # 'n_comp' must be an integer in [1,ndim[,  'theta0' a list of n_comp values

    t = GEKPLS(n_comp=2, theta0=[1e-2, 1e-2], xlimits=funXLimits, delta_x=1e-2, extra_points=1, print_prediction=False)
    t.set_training_values(xt, yt)
    # Add the gradient information
    for i in range(ndim):
        t.set_training_derivatives(xt, yt[:, 1 + i].reshape((yt.shape[0], 1)), i)
    t.train()

    # Prediction of the validation points
    y = t.predict_values(xtest)
    print('GEKPLS1,  err: ' + str(compute_rms_error(t, xtest, ytest)))
    title=('GEKPLS model')
    return t, title, xtest, ytest

    # Prediction of the derivatives with regards to each direction space
    yd_prediction = np.zeros((ntest, ndim))
    for i in range(ndim):
        yd_prediction[:, i] = t.predict_derivatives(xtest, kx=i).T
        print(
            'GEKPLS1, err of the ' + str(i + 1) + '-th derivative: ' + str(compute_rms_error(t, xtest, ydtest[:, i], kx=i)))

        if plot_status:
            plt.plot(ydtest[:, i], ydtest[:, i], '-.')
            plt.plot(ydtest[:, i], yd_prediction[:, i], '.')

        if plot_status:
            plt.show()




