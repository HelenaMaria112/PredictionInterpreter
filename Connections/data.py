#import pandas as pd
#import jaydebeapi as jdbc  # also use older version of JPype: pip install JPype1==0.6.3 --force-reinstall
import datetime
def getData():
    # corpus to add Datasets. this method is overwritten in PredictionInterpreter.py
    '''
    try:
        # connect to database
        jdbc_class = "com.intersystems.jdbc.IRISDriver"
        url = "jdbc:XX://127.0.0.1:1972/XX"
        jdbc_path = "C:/InterSystems/Cache/dev/java/lib/JDK18/intersystems-jdbc-3.0.0.jar"
        conn = jdbc.connect(jclassname=jdbc_class,
                            url=url,
                            jars=jdbc_path,
                            libs=jdbc_path)
        print('Connecting to {}'.format(url))
        # Get data
        sql = "select * from ki.ifilter_fibu"
        df = pd.read_sql(sql, conn)
        conn.close()
    except Exception as e:
        print("Error: {}".format(str(e)))
        # sys.exit(1)
        df = 1
    return pd.DataFrame(df)
    '''

def saveDataAnalysis(data):

    for dataColumn in data.columns:
        analysis = data.groupby(dataColumn).count()
        ystr = dataColumn + "\n"
        for elementIndex in range(len(analysis)):
            ystr = ystr + analysis.index[elementIndex] + "\t"
            ystr = ystr + str(analysis.iloc[elementIndex, 0]) + "\n"
            if elementIndex % 10 == 0 and elementIndex != 0:
                ystr = ystr + "\n"
        docName = dataColumn + "analysisData" + str(datetime.datetime.now())[:19].replace(":", "-") + ".txt"
        f = open(docName, "w+")
        f.write(ystr)
        f.close()
    print("all files saved")