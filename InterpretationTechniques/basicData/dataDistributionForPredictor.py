import datetime
import numpy as np
def writeDistribution(data, groupbyColumn):
    '''
    :param data: pandas dataframe with datasets where each row represents a dataset
    :return: writes file with summary of data (unique values per column)
    '''
    for column in data.columns:
        print("uniqueness of " + str(column) + "   " + str(
            len(data[column].unique()) == len(data[column])))
    print("lenght of data: " + str(len(data)))

    ystr="Data distribution   \n "
    for column in data.columns:
        columnDistribution = data.groupby(groupbyColumn).nunique()

        ystr = ystr + str(column) + " \n " + " \t "
        for column in columnDistribution:
            ystr = ystr + column + " \t "
        ystr = ystr + " \n "

        for rowIndex in range(len(columnDistribution)):
            ystr = ystr + columnDistribution.index.values[rowIndex] + " \t "
            for columnIndex in range(len(columnDistribution.columns)):
                ystr = ystr + str(columnDistribution.iloc[rowIndex, columnIndex]) + " \t "
            ystr = ystr + " \n "
            if rowIndex % 10 == 0 and rowIndex != 0:
                ystr = ystr + " \n "
        ystr = ystr + " \n \n \n "

        docName = "dataInformationDistribution.txt" + str(datetime.datetime.now())[:19].replace(":", "-") + ".txt"
        f = open(docName, "w+")
        f.write(ystr)
        f.close()