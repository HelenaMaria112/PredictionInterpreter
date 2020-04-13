import pandas as pd
from random import randrange

class UniqueData:
    clientTable = pd.Series()  # saves all inserted client values
    seldomDataInserted = False  # indicates whether all seldomData is inserted already
    seldomDataIndex = 0

    def getUniqueData(self, data, amountOfDatasets, timesClientCanBeUsed, seldomDataInserted = False):
        self.seldomDataInserted  = seldomDataInserted
        self.clientTable = pd.Series()
        self.timesClientCanBeUsed = timesClientCanBeUsed
        self.data = data
        self.selectedData = pd.DataFrame(columns=data.columns)
        self.seldomData = self.getSeldomData(data)  # datasets that contain values that are seldomly used

        #append more
        self.dataIndexes = [0] * len(data)
        while min(self.dataIndexes) == 0 and (self.selectedData.ndim == 1 or len(self.selectedData) < amountOfDatasets):
            dataset = self.getNextDataset()
            self.selectedData = self.selectedData.append(self.checkAndInsert(dataset))

        self.printStatistics()
        return self.selectedData

    def printStatistics(self):
        print(self.clientTable)
        for column in self.data.columns:
            print("uniqueness of " + str(column) + "   " + str(len(self.selectedData[column].unique()) == len(self.selectedData[column])))
        print("lenght of data: " + str(len(self.selectedData)))
        column = "client"

        print("client usage: max: " + str(max(self.selectedData.groupby(column).count())) +  "min:  " + str(min(self.selectedData.groupby(column).count())))
        print(self.selectedData.groupby(column).count())

    def checkAndInsert(self, dataset):
        # check if insertion posible and insert
        client = dataset['client']
        datasetInsertion = None
        if client not in self.clientTable.index.tolist() or self.clientTable[client] < self.timesClientCanBeUsed:
            if self.noDoublesInDataWithNewDataset(dataset):
                self.updateClientTable(client)
                datasetInsertion = dataset
        return datasetInsertion

    def noDoublesInDataWithNewDataset(self, dataset):
        for column in self.data.columns.drop('client'):
            if self.selectedData.ndim == 1: #only one dataset
                columnValuesInData = self.selectedData[column]
            else:
                columnValuesInData = self.selectedData[column].values
            if dataset[column] in columnValuesInData:
                return False
        return True

    def updateClientTable(self, client):
        if client in self.clientTable.index:
            self.clientTable[client] += 1
        else:
            self.clientTable[client] = 1

    def getNextDataset(self):
        # take seldom data first, then other data
        dataset = None
        if self.seldomDataInserted:
            dataset = self.getNext(self.data)
        else:
            dataset = self.seldomData.iloc[self.seldomDataIndex]
            self.seldomDataIndex += 1
            if self.seldomDataIndex == len(self.seldomData):
                self.seldomDataInserted = True
        return dataset

    def getSeldomData(self, data):
        seldomData = pd.DataFrame(columns= data.columns)   # datasets that contain seldom values are inserted first to asure that these values are represented
        seldomData = seldomData.append(self.datasetsLessThan(data, ['client'], 2))
        seldomData = seldomData.append(self.datasetsLessThan(data, ['client'], 7))
        for column in data.columns:
            seldomData = seldomData.append(self.datasetsLessThan(data, [column], 4))
        seldomData = seldomData.append(self.datasetsLessThan(data, ['konto'], 7))
        seldomData = seldomData.append(self.datasetsLessThan(data, data.columns.to_list(), 26))
        return seldomData.drop_duplicates()

    def getNext(self, data):
        newIndex = randrange(len(data)-1)
        origNewIndex = newIndex

        while True:
            if self.dataIndexes[newIndex] == 0:
                self.dataIndexes[newIndex] = 1
                return data.iloc[newIndex]
            newIndex += 1
            if newIndex == len(data):
                newIndex = 0
            if newIndex == origNewIndex:
                print("all data tested. No matching subset found")
                return data.iloc[origNewIndex]    #just return any dataset. it'll be too much eather way

    def ValuesLessThan(self, data, column, lessThanValue):
        df = data
        # v = df[['Col2', 'Col3']]
        v = data[['client']]
        return df[v.replace(v.stack().value_counts()).lt(lessThanValue).all(1)].loc[:]['client']

    def datasetsLessThan(self, data, column, lessThanValue):
        '''
        :param data:
        :param column:
        :param lessThanValue:
        :return:
            datasets from data which values of a column are represented less than x times
        '''
        df = data
        # v = df[['Col2', 'Col3']]
        v = data.loc[:][column]
        # delete all rows that are represented more often but lessThanValue
        return df[v.replace(v.stack().value_counts()).lt(lessThanValue).all(1)]

def getUniqueData(data):
    data1 = data
    while True:
        ud = UniqueData()
        timesClient=1
        data1 = ud.getUniqueData(data, 26,timesClient, seldomDataInserted=True)
        if len(data1) == 9*timesClient:
            break
    return data1