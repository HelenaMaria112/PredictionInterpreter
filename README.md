# Prediction Interpreter

This package contains a wrapper to interpret any predictor. Use this package:
 - to interpret a machine learning model that runs over a service
 - to interpret a trained ML- model in python, that has insufficient features for interpretation 
 - to interpret a trained  ML- model in python, using the methods already set up for you
 - to get ideas about how to use the interpretation methods
 
By using established interpretation- techniques, the package only needs a function that takes a dataset and returns a prediction, as well as some information about the data and columns.
The package then automatically sets up a wrapper containing the important information and configurations to run interpretations.

To set up the predictor, state the necessary information as in file StartPredictionInterpreter given:
Here, we use a DummyML-Model that predicts the amount of visitors on a day, depending on whether it is holiday, how much the ticket price is, and what'S the weather like.
(For your additional information: There are many visitors if its holiday, or when the price is low. The weather has a random effect on the amount of visitors. But all of this you will see when running the dummy data.) 

Additionally to the DummyMLModel, it is necessary to give 
 - the testdata as a panda-Dataframe
 - as well as the result column's name
 - all data columns' names
 - the numerical column names (if there are any...)
 - the categories in result column (classes_)
 - rather the result is a continuous value
  
 
     #get data and object that has singlepredict in correct format
    dm = DummyMLModel()
    data = dm.testdata

    #define necessary variables for techniques
    standardColumns = data.columns.to_list()
    resultcolumn = "visitorsOnThisDay"
    listOfNumericalColumns = ["ticketPrice"]
    _classes_ = data[resultcolumn].unique().tolist()
    resultIsContinuous = False

After that, create the the interpreter with the before defined parameters, and run the interpretation techniques that are of interest to you. 

    #create interpreter 
    predictionInterpreter = PredictionInterpreterClass(dm.predict, listOfNumericalColumns, standardColumns, resultcolumn, _classes_, data, resultIsContinuous)
    
    #call interpretation techniques you want to use:
    predictionInterpreter.plotpdpOfDistanceToTrueResultSklearn() # only works if called without any prior methods
    predictionInterpreter.plotpdpOfDistanceToTrueResultSklearn2D()
    predictionInterpreter.writeDistribution("visitorsOnThisDay")
    predictionInterpreter.plotConfusionTable()
    predictionInterpreter.printImportanceEli5(exceptedColumns = resultcolumn)
    predictionInterpreter.printImportanceEli5(distanceAnalysis=True)
    predictionInterpreter.featureAnnulation(annulationValue = "0")
    predictionInterpreter.plotIce()
    predictionInterpreter.plotpdpOfDistanceToTrueResultPdpbox(featureToExamine="ticketPrice")
    predictionInterpreter.plotpdpOfDistanceToTrueResultPdpbox(featuresToExamine=["holidayYN", "ticketPrice"])
    predictionInterpreter.plotpdpOfDistanceToTrueResultPdpbox(featureToExamine="ticketPrice", featuresToExamine=["holidayYN", "ticketPrice"])
    predictionInterpreter.globalSurrogateModel()