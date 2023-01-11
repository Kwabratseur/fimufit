from mufit import *
import pytest

def setupLinear(a=33.298490,b=-7.5483425):
    x = [i for i in range(100)]
    y = [a*i+b for i in x]
    df = pd.DataFrame([y,x]).T
    df.columns=["y","x"]
    return df

@pytest.fixture
def TI():
    lambdaMap = {"y": ["x", "$a", "$b"]}
    lambdaDict = {"y": lambda x, a, b: (x*a)+b}
    a = MultiFitter(lambdaMap = lambdaMap, lambdaDict = lambdaDict)
    a.dt = 1
    return a

@pytest.fixture
def TI2():
    a = MultiFitter(verbose=1)
    a.loadData(SimThis,initPredictors=True)
    return a
df = setupLinear()
#TI = setupTestInstance(dflin)


def test_Loading(TI):
    TI.loadData(df,initPredictors=True)
    assert TI.DF.equals(df), "FAIL input df does not matches output"
    print("PASS data loading")
    for i in TI.predictors:
        assert TI.predictors[i] is not None, "FAIL predictor {} not properly set from DF!".format(i)
    print("PASS predictor initialization")
    constants = {"a":33.3,"b":-7.54}
    TI.loadConstants(constDict = constants)
    constants = TI.constants
    assert TI.constants["a"] == constants["a"], "FAIL Constant a not loaded properly"
    assert TI.constants["b"] == constants["b"], "FAIL Constant b not loaded properly"
    print("PASS constant loading")


def test_Evolving(TI):
    TI.loadData(df,initPredictors=True)
    init = TI.Simulate(1,err="y") # serves to initialize error frame, 1 iteration
    assert len(init) == 1, "FAIL simulate makes more frames then expected! {}".format(len(init))
    print("PASS simulate initialization")
    constants = TI.constants
    output = TI.Evolver("y",N=100) # for each  parameter 100 iterations; so 200 + 2 init + what came before expected error length
    assert len(output) == 100*2 + 1, "FAIL! length of evolver does not equal intention! {}".format(len(output))
    print("PASS evolver output length")
    for i in constants:
        assert TI.solveList[i] != constants[i], "FAIL constant {} did not change after 100 evolution steps".format(i)
    print("PASS evolver changing")

def test_SearchMap(TI):
    TI.loadData(df,initPredictors=True)
    #TI.error = []
    SMap = {"a":[1.0,100.0],"b":[-10.0,10.0]}
    constants = TI.constants
    errors = TI.parameterSearchMap(constraints=SMap,err="y",ims=4,steps=25)
    assert len(errors) == 25*4*4*2, "FAIL not expected searchmap space {}".format(len(errors))
    print("PASS parameterSearchMap looks in correct solution space")
    assert hasattr(TI,"pSearchMap"), "FAIL pSearchMap not internally defined!"
    for i in TI.pSearchMap:
        assert len(TI.pSearchMap) == 4*4, "FAIL amount of generated intermediates is not correct!{}".format(len(TI.pSearchMap))
    print("PASS searchmap is correctly initialized")


def test_analyzeErrors(TI):
    TI.loadData(df,initPredictors=True)
    init = TI.Simulate(1,err="y") # serves to initialize error frame, 1 iteration
    output = TI.Evolver("y",N=100)
    errors = TI.AnalyzeErrors()
    assert len(errors) == TI.dataLength, "FAIL AnalysError does not expected amount of errors {}".format(len(errors))
    print("PASS datalength analyzeErrors")
    assert errors[0][1] < errors [-1][1], "FAIL Errors were not sorted out properly"
    print("PASS sorting analyzeErrors")
    for i in errors:
        print(i)

def test_AnalyzeBulkEvolver(TI2):
    SMap = {"U":[0.01,50.0],"C":[10,2000]}
    #error = TI2.BulkError([25.0,1200],"Tobj")
    #assert type(error) == type(100.0), "FAIL, BulkError does not return a float!:{}".format(error)
    errors = TI2.BulkEvolver("Tobj",constraints=SMap)
    assert errors[0][0] < errors[-1][0],"FAIL, BulkEvolver did not optimize errors!:{}".format(errors)
    error = TI2.BulkError([25.0,1200],"Tobj")
    assert type(error) == type(100.0), "FAIL, BulkError does not return a float!:{}".form
