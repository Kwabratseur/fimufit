#!/usr/bin/env python3
"""! @brief Defines the MultiFitter classes.

  @file mufit.py

  @brief Defines the MultiFitter class.

  @section description_MultiFitter Description
  Defines the base and end user class for all fitting purposes, model and data agnostic
  - multifitter

  @section libraries_MultiFitter Libraries/Modules
  - random standard library (https://docs.python.org/3/library/random.html)
    - Access to randint function.
  - pandas dataframe library
  - datetime standard library
  - matplotlib plotting Library
  - plotly plotting Library
  - itertools standard library
  - seaborn heatmap library

  @section notes_MultiFitter Notes
  - Comments are Doxygen compatible.

  @section todo_MultiFitter TODO
  - many.
  - setting best parameters for instance

  @section author_sensors Author(s)
  - Created by Jeroen van 't Ende 27/11/2022

  Copyright (c) 2025 Jeroen van 't Ende.  All rights reserved.
"""
import pandas as pd # dataframe module, allows for easy measurement handling and resampling
from datetime import datetime # framework to interpret time
import plotly.graph_objects as go # interactive plotting framework
import matplotlib.pyplot as plt # non-interactive plotting framework
from scipy.optimize import minimize, least_squares # scientific python optimization function
#pd.options.plotting.backend = "plotly" # set interactive plotting framework as default to use, enables df.plot() or df[["selection1","selection2","etc"]].plot()
import random, itertools
import seaborn as sn
import os
import json


## Example profile dictionary as input for the generator
Profile = {"Tenv":[10,20,18,40,25,60,18,80,8,100],
           "Pe":[700,10,200,20,0,60,700,100], # 700W power up to point 10, then 200W up to point 20, then 0 W up to 60 and finally 700W up to point 100
           "Tobj":[24,100],
           "Psol":[0,10,100,20,200,30,300,40,250,50,225,60,200,70,175,80,125,90,0,100]} # 24C indoor from 0 to 100

def ProfileGenerator(length,Profile):
    """! Profile generator function, returns a dataframe with the profile.

    @param  length  the length of the dataset to be generated
    @param Profile  A dictionary with name:array pairs where the array has format [value,upto,value,upto,etc....]
    @param verbose  verbose boolean generates more output for debugging

    @return A Pandas dataFrame of length with the specified profile
    """
    df = pd.DataFrame(index=range(length)) # instantiate dataframe
    for i,x in enumerate(Profile): # for all values that need to be put in column, in this case Ta, Pe, Ti
        for j,y in enumerate(Profile[x]): # for all profile points in Ta, Pe, Ti
            if(j==0): #if it is the first element, fill from element 0 up to the next change
                df.loc[0:Profile[x][j+1],(x)] = y # create zero column to contain prediction
            if(j%2==0): # if modulus 2 of index is true; aka even numbered indexes
                df.loc[Profile[x][j-1]:Profile[x][j+1],(x)] = y # fill from last change up to this change with fixed value y
    return df

## An Example generated profile
SimThis = ProfileGenerator(100,Profile) # we give it the length of the set and the profile to generate.

class MultiFitter(object):
    """! The MultiFitter base class.

    Defines the base class which enables parameter fitting, loading data, visualization, simulation
    """
    def __init__(self,Instance,lambdaMap=None,lambdaDict=None,verbose=0):
        """! The multifitter base class initializer.

        @param lambdaMap  lambdaMap dictionary with array of variables, constants have a prefix of $ when they have to be fitted, * if they are truly constant
        @param lambdaDict  lambdaDict dictionary with lambdafunctions
        @param verbose  verbose boolean generates more output for debugging

        @return  An instance of the Sensor class initialized with the specified name.
        """
        ## The verbose boolean
        self.Verbose = verbose
        if lambdaMap is not None:
            self.lambdaMap = lambdaMap
        else:
            if self.Verbose >= 1:
                print("lambdaMap not initialized, using testcase")
            self.lambdaMap = {"Qtr": ["Tobj","Tenv","$U"],
                              "Qsol":["Psol"],
                              "Qel":["Pe"],
                              "dTobj":["Qtr","Qsol","Qel","dt","$C"],
                              "Tobj":["Tobj","dTobj"]}
        if lambdaDict is not None:
            self.lambdaDict = lambdaDict
        else:
            if self.Verbose >= 1:
                print("lambdaDict not initialized, using testcase")
            self.lambdaDict = {"Qtr": lambda Tobj, Tenv, U: (Tenv-Tobj)*0.001/ U,
                               "Qsol": lambda Psol: Psol*0.001,
                               "Qel": lambda Pe: Pe*0.001,
                               "dTobj": lambda Qtr, Qsol, Qel, dt, C: (Qtr+Qsol+Qel)*dt/C,
                               "Tobj": lambda Tobj, dTobj: Tobj + dTobj}
        ## The variables change every step
        self.variables = {"dt": 1.0}
        ## The constants stay constant unless fitted
        self.constants = {}
        ## The solvelist so that not all constants have to be fitted
        self.solveList = {}
        ## The predictors calculated by applying lambdafunctions
        self.predictors = {}
        ## The error array contains [index, error, [constants]]
        self.index = 0
        self.error = []
        self.dt = 1
        self.Instance = "{}.json".format(Instance)
        for i in self.lambdaMap:
            for j in self.lambdaMap[i]:
                if "*" in j:
                    self.constants[j[1:]] = 1.0
                elif "$" in j:
                    self.constants[j[1:]] = 1.0
                    self.solveList[j[1:]] = 1.0
                else:
                    if j not in self.lambdaMap:
                        self.variables[j] = 1.0
            self.predictors[i] = 1.0

        columns = ["index","error","y","y-predicted"]
        for i in self.constants:
            columns.append(i)
        self.columns = columns

        Files = os.listdir()
        if self.Instance in Files:
            if self.Verbose >= 1:
                print("Stored data found..")
            self.LoadedInstance = self.storeJson()
            self.loadConstants(self.LoadedInstance)
        else:
            if self.Verbose >= 1:
                print("No data stored yet.. Creating file")
            self.storeJson(data=self.constants)

        if self.Verbose >= 1:
            print(self.solveList)
            print(self.constants)
            print(self.variables)
            print(self.predictors)

    def storeJson(self,data=None):
        if data is not None:
            with open(self.Instance, 'w') as f:
                json.dump(data, f)
                f.close()
            datafmt = {"date":str(datetime.now()), "index":self.index, "constants":data}
            with open("Backup_constants.json", "a") as f:
                json.dump(datafmt, f)
                f.close()
        else:
            with open(self.Instance, 'r') as f:
                data = json.load(f)
                f.close()
            return data

    def parameters(self):
        """! Retrieves all current variables, constants and predictors

        @return  A dictionary with variables, constants and predictors
        """
        res = {**self.variables, **self.constants, **self.predictors}
        return res

    def Calculate(self):
        """! Retrieves all current variables, constants and predictors,
            filles them into lambdafunctions and puts the results into predictors
        """
        for i in self.lambdaDict:
            args = []
            if self.Verbose >= 5:
                print("Calculating {}, gathering inputs {}".format(i,self.lambdaMap[i]))
            for j in self.lambdaMap[i]:
                if "$" in j:
                    j = j[1:]
                if "*" in j:
                    j = j[1:]
                args.append(self.parameters()[j])
            self.predictors[i] = self.lambdaDict[i](*args)
            if self.Verbose >= 5:
                print("input {} gave {}".format(args,self.predictors[i]))

    def loadVariables(self,varDict = None):
        """! Load a dictionary of variables and add in dt

        @param varDict  dictionary with new variables and values
        """
        self.variables["dt"] = self.dt
        if varDict is not None:
            if self.Verbose >= 4:
                print("Loading new variables: {}".format(varDict))
            for i in varDict:
                self.variables[i] = varDict[i]
        else:
            print("Error, expecting a dict in format: {}".format(self.variables))

    def loadConstants(self,constDict = None):
        """! Load a dictionary of constants and update solvelist too

        @param constDict  dictionary with new constants and values
        """
        if constDict is not None:
            if self.Verbose >= 4:
                print("Loading new constants: {}".format(constDict))
            if constDict == "best":
                self.LoadedInstance = self.storeJson()
                for i in self.LoadedInstance:
                    self.constants[i] = self.LoadedInstance[i]
                    if i in self.solveList:
                        self.solveList[i] = self.LoadedInstance[i]
                if self.Verbose >= 4:
                    print("Loaded stored params: {}".format(self.constants))
            else:
                for i in constDict:
                    self.constants[i] = constDict[i]
                    if i in self.solveList:
                        self.solveList[i] = constDict[i]
        else:
            print("Error, expecting a dict in format: {}".format(self.constants))

    def loadPredictors(self,constDict = None):
        """! Load a dictionary of predictors

        @param constDict  a dictionary with new predictors
        """
        if constDict is not None:
            if self.Verbose >= 4:
                print("Loading new constants: {}".format(constDict))
            for i in constDict:
                if i in self.predictors:
                    self.predictors[i] = constDict[i]
        else:
            print("Error, expecting a dict in format: {}".format(self.constants))

    def ArrayToConstants(self,Array):
        """! load an array of constants; expects same length and order as internal dictionary!


        @param Array  An array formatted to the same format as self.Constants
        """
        copy = self.constants.copy()
        for i,x in enumerate(self.solveList):
            try:
                copy[x] = Array[i]
            except IndexError:
                if self.Verbose >= 2:
                    print("{} : index {} not defined in Array, not solved for!".format(x,i))
        self.loadConstants(copy)

    def dictToArray(self,Dict):
        """! convert dictionary to array

        @param Dict  dictionary to convert

        @return  An array with the dictionaries values
        """
        return [Dict[i] for i in Dict]

    def dictSlice(self,index):
        """! return a dictionary slice from the loaded dataframe

        @param index  index from DF to return

        @return  A dictionary with parameters found at DF[index]
        """
        return self.DF.iloc[index].to_dict()

    def writeSlice(self):
        """! write the current predictors to SDF simulation dataframe
        """
        #self.SDF[index] = pd.Series(self.predictors).T
        TempDict = self.predictors.copy()
        self.SDF = pd.concat([self.SDF,pd.Series(self.predictors).to_frame().T], ignore_index=True)

    def resetIndex(self):
        """! reset the index to 0 and reset predictors and variables from SDF[0] IF SDF is defined
        """
        self.index = 0
        if hasattr(self, 'SDF'):
            self.loadPredictors(self.SDF.iloc[self.index].to_dict())
            self.loadVariables(self.SDF.iloc[self.index].to_dict())

    def scipyOptimize(self,constraints=None, err=None, steps = None, iterations=1000, solver=None, lsq = False):
        """! Do parameter optimization with scipy.optimize

        @param constraints  dictionary of arrays with [min,max] values to constraint the parameter search to
                            example: SMap = {"U":[0.01,45.0],"C":[100,200]}. If not given, all in solveList will be optimized without constraints
        @param err  Name of variable to track as error, make sure this is in the predictors and in the loaded data
        @param steps  amount of dataset steps to calculate in the error function
        @param iterations  amount of allowed solver iterations, default set to 1000
        @param solver  solving algorithm to be used, default nelder-mead
        @param lsq  treat problem as least-squares problem

        @return  An array with the best parameters found
        """
        if err is not None:
            x0 = []
            for i in self.solveList:
                x0.append(self.solveList[i])
            if constraints is not None:
                bounds = []
                for i in constraints:
                    bounds.append(constraints[i])
                if self.Verbose >= 1:
                    print(bounds)
                    print(self.solveList)
                if lsq == True:
                    result = least_squares(fun=self.BulkError, x0=x0, args=(err,), bounds=bounds, max_nfev = iterations)
                else:
                    result = minimize(fun=self.BulkError, x0=x0, args=(err,), bounds=bounds, tol=1e-6, method=solver, options={'maxiter':iterations})
            else:
                if lsq == True:
                    result = least_squares(fun=self.BulkError, x0=x0, args=(err,), max_nfev = iterations)
                else:
                    result = minimize(fun=self.BulkError, x0=x0, args=(err,),  tol=1e-6, method=solver, options={'maxiter':iterations})
        return result


    def BulkError(self,params,err, Iterations=None):
        """! Do optimization run for scipy solver

        @param params: parameter array, generated by scipy.
        @param err: error objective

        @return  The accumulated error for the last simulation run
        """
        if self.Verbose >= 3:
            print(params, err)
        for i,x in enumerate(self.solveList):
            self.constants[x] = params[i]
            self.solveList[x] = params[i]
        if Iterations is None:
            self.resetIndex()
            errors = self.Simulate(int(self.dataLength),err=err)[-self.dataLength:]
        else:
            errors = self.Simulate(int(Iterations),err=err)[-Iterations:]
        accerr = 0
        for i in errors:
            accerr += i[1]
        return accerr


    def parameterSearchMap(self,constraints=None, err=None, ims = 4, steps = None):# format of "coef":[min,max], etc
        """! Do parameter search over a defined area, results in steps*(constants^ims) iterations

        @param constraints  dictionary of arrays with [min,max] values to constraint the parameter search to
                            example: SMap = {"U":[0.01,45.0],"C":[100,200]}
        @param err  Name of variable to track as error, make sure this is in the predictors and in the loaded data
        @param ims  Intermediate steps to generate, if 0 and 1 are given, a parameter search for 0, 0.25, 0.5 and 1 will be executed

        @return  An array with errors in the format of [index,error,c0,c1,cn....]
        """
        if constraints is not None and err is not None:
            self.constraints = constraints
            constraintArray = []
            if self.Verbose >= 1:
                print("Constraints {} were provided".format(constraints))
            for i in constraints:
                Min = constraints[i][0]
                Max = constraints[i][1]
                Delta = Max-Min
                for j in range(ims-2):
                    constraints[i].append(Min+((Delta/ims)*(j+1)))
                constraints[i] = sorted(constraints[i])
                constraintArray.append(constraints[i])
                if self.Verbose >= 1:
                    print("{} will be tested for {}".format(constraints[i],i))
            pSearchMap = []
            for i in itertools.product(*constraintArray):
                pSearchMap.append(i)
            if self.Verbose >= 1:
                print("{} parameter sets were generated and will be tested".format(len(pSearchMap)))
            self.pSearchMap = pSearchMap
            for i in self.pSearchMap:
                for j,x in enumerate(self.constraints):
                    self.constants[x] = i[j]
                    if steps is None:
                        steps = self.dataLength
                    self.Simulate(steps,err=err)
            return self.error
        else:
            print("Provide constraints!")




    def Fiddler(self,curr,previousR,currentR,constantIndex): # wqe want to compare the previous error and previous previous error, to at least determine the best "direction"
        """! Parameter transformer/gradient descent algorithm

        @param curr  current parameter that will be transformed, change is proportional to this.
        @param previousR  previous error array, this is used to determine the best direction to change towards
        @param currentR  current error array, as mentioned, to implement some form of gradient descent
        @param constantIndex  index of the current constant that is modified, needed to know what parameter in error array to look at

        @return  A new constant to try
        """
        sign = random.randint(0,10)
        if self.Verbose >= 5:
            print("###########-- Fiddler input: i:{}, current:{} - {} previous:{}".format(constantIndex,curr,currentR,previousR))
        if previousR[1] < currentR[1]: #
            if self.Verbose >= 5:
                print("Previous set of variables better")
            if previousR[constantIndex] < currentR[constantIndex]:
                if sign > 9:
                    sign = 1
                else:
                    sign = -1
            else:
                if sign > 9:
                    sign = -1
                else:
                    sign = 1
        else:

            if sign > 5:
                sign = -1
            else:
                sign = 1
        Amount = (random.randint(1,1000)/999)*0.1
        return curr + (curr*Amount*sign)


    def BulkEvolver(self, err, constraints = None, N=10, repeats=25, Iterations = None):
        self.Simulate(1,err=err)
        ##Instantiation of errorlist
        Errorlist = []
        params = self.dictToArray(self.solveList)
        error = self.BulkError(params,err, Iterations=Iterations)
        Errorlist.append([error])
        for i in self.solveList:
            Errorlist[-1].append(self.solveList[i])
        ##Instantiation of errorlist
        for k in range(repeats):
            for constantIndex,j in enumerate(self.solveList):
                Errorlist.sort()
                if self.Verbose >= 2:
                    print("Best error: {}-{} - {}".format(k,j,Errorlist[0]))
                p0 = Errorlist[0][constantIndex+1] # get best result so far for next optimization run; thats why we need to run before looping
                p1 = p0
                err0 = Errorlist[0][0] # added benefit, instantiate a better error to compare with!
                for i in range(N):
                    params = self.dictToArray(self.solveList)
                    p1 = params[constantIndex]
                    error = self.BulkError(params,err, Iterations=Iterations)
                    Errorlist.append([error])
                    for i in self.solveList:
                        Errorlist[-1].append(self.solveList[i])
                    p0 = self.Fiddler(p1,[0]+Errorlist[-2],[0]+Errorlist[-1],2+constantIndex)
                    if constraints is not None:
                        if p0 < constraints[j][0]:
                            if self.Verbose >= 1:
                                print("Constraining down {}:{} < {}".format(j,p0,constraints[j][0]))
                            p0 = constraints[j][0] + abs(constraints[j][0]*0.2)
                        elif p0 > constraints[j][1]:
                            if self.Verbose >= 1:
                                print("Constraining up {}:{} > {}".format(j,p0,constraints[j][1]))
                            p0 = constraints[j][1] - abs(constraints[j][1]*0.2)
                    self.solveList[j] = p0
                    err0 = Errorlist[-2]
        Errorlist.sort()
        self.ArrayToConstants(Errorlist[0][1:])
        self.storeJson(self.constants)
        if self.Verbose >= 1:
            print("Best parameters found so far: {}".format(Errorlist[0]))
        return Errorlist


    def Evolver(self,err,N = 10):
        """! Evolution inspired parameter search algorithm

        @param err  name of predictor (and dataframe column) that will be used for error determination; prediction target.
        @param N  amount of iteration to optimize each parameter that is in the solveList

        @return  An array with errors in the format of [index,error,c0,c1,cn....]
        """
        try:
            ROld = self.error[-1].copy()
        except:
            self.Simulate(1,err)
            ROld = self.error[-1].copy()
        varstore = self.variables.copy()
        indexstore = self.index
        predictorstore = self.predictors.copy()
        for constantIndex,j in enumerate(self.solveList):
            p0 = self.solveList[j]
            R = self.error[-1].copy()
            ROld = R
            pOld = p0
            if self.Verbose >= 1:
                print("----------------- Solving for {}:{} - p0:{} pOld:{}".format(constantIndex,j,p0,pOld))
            for i in range(N):
                pOld = p0
                ROld = self.error[-1].copy()
                p0 = self.Fiddler(pOld,ROld,R,4+constantIndex)
                self.constants[j] = p0
                R = self.Simulate(1,err=err,write=False)[-1]
                # self.variables = varstore.copy()
                # self.index = indexstore
                # self.predictors = predictorstore.copy()
                if self.Verbose >= 2:
                    print("i:{} Old R:{} P:{} - new R:{} P:{} - {}".format(self.index,ROld,pOld,R,p0,self.predictors))
                if R[1] < ROld[1]:
                    if self.Verbose >= 2:
                        print("Found better error {}:{} (old:{}) for Ts={}".format(j,p0,pOld,self.index))
                    self.solveList[j] = p0
                else:
                    #R = ROld
                    p0 = pOld
        # self.variables = varstore.copy()
        # self.index = indexstore
        # self.predictors = predictorstore.copy()
        # self.loadConstants(self.solveList)
        return self.error

    def stepIndex(self):
        """! Step through the index and reset it once datalength is exceeded
        """
        self.index += 1
        if self.index > self.dataLength:
            self.resetIndex()

    def loadData(self,df,initPredictors=True):
        """! Load pandas dataframe for simulation/prediction/parameter fitting

        @param df  pandas dataframe to load, this will also define SDF and datalength
        @param initPredictors  Boolean which fills the predictor dictionary with initial values from the dataframe
        """
        ## The loaded pandas dataFrame
        self.DF = df
        if hasattr(self,"SDF"):
            del self.SDF
        ## The loaded dataFrame length
        self.dataLength = len(df)-1
        self.resetIndex()
        ## The Simulated model, predictor values are written to here
        self.SDF = pd.DataFrame(columns=[i for i in self.predictors])
        if self.Verbose >= 1:
            print(self.SDF)
        if initPredictors:# if predictors are found in the df; initialize first point.
            for i in self.predictors:
                if i in df.columns:
                    self.predictors[i] = df[i].iloc[0]

    def resampleData(self,timeStep): # timestep in seconds
        """! Internal dataframe resampler; this is not a smart function! make sure data is time dependent.

        @param timeStep  Resample step in seconds; only works when the loaded data uses a datetime index!
        """
        if self.Verbose >= 1:
            print(len(self.DF))
        self.DF = self.DF.resample("{}s".format(timeStep)).interpolate().bfill().ffill()
        self.dt = timeStep
        self.dataLength = len(self.DF)-1
        if self.Verbose >= 1:
            print(len(self.DF))

    def AnalyzeErrors(self): # go through self.errors; for each index: find best set of parameters
        """! Error analysis and sorting function, puts best found parameters in constants

        @return  An array with errors in the format of [index,error,yreal,ypred,c0,c1,cn....]
        """
        errors = []
        for i in range(self.dataLength):
            lowest = 10000000000
            #errors.append(self.error[0])
            for x in self.error:
                if x[0] == i: #this is the index we're looking for
                    if lowest == 10000000000:
                        errors.append(x)
                        lowest = 1000000000
                    if x[1] < lowest:
                        errors[-1][1] = x[1]
                        lowest = errors[-1][1]
        avgs = [1.0 for i in errors[-1]]
        for i in errors:
            for j,x in enumerate(avgs):
                avgs[j] = (i[j]+x)/2
        self.storeJson(self.constants)
        if self.Verbose >= 1:
            print("best constants, lowest error (index):{}".format(self.constants))
        return errors

    def plotError(self,All=False):
        """! Plots the internal error array of shape [index, error, c0, c1, cn....]

        @param All  if set to true, all errors are shown. Otherwise only best parameters for each timestep. Plotly plot
        """
        df = pd.DataFrame(self.AnalyzeErrors(),columns=self.columns)
        if All:
            df = pd.DataFrame(self.error,columns=self.columns)
        for i in self.solveList:
            fig = go.Figure(data=[go.Scatter3d(x=df["index"], y=df["error"], z=df[i],mode='lines')])
            fig.update_layout(scene = dict(
                    xaxis_title='index',
                    yaxis_title='error',
                    zaxis_title=i),
                    width=900,
                    margin=dict(r=20, b=10, l=10, t=10))
            fig.write_html("Mufit_Error_{}.html".format(i), auto_open=False)
            fig.show()

    def setSolveList(self,Array=None,Dict=None):
        """! Set the constants to solve, can take an array of indexes/names or dict with new values

        @param Array  An array with indexes or string names which can be found in constants and should be solved
        @param Dict  A dictionary with key:value pairs where key=name and value = initial value
        """
        if Array is not None:
            idx = False
            NewDict = {}
            if type(Array[0]) == type(10):
                idx = True
            for i, x in enumerate(self.constants):
                if idx == True:
                    if i in Array:
                        NewDict[x] = self.constants[x]
                else:
                    if x in Array:
                        NewDict[x] = self.constants[x]
            self.solveList = NewDict.copy()
        elif Dict is not None:
            self.solveList = Dict.copy()

    def Simulate(self,steps,err=None,write=True):
        """! Apply the model (Calculate) to the loaded data, does not strictly need a DF

        @param steps  amount of steps to simulate
        @param err  name of predictor (and dataframe column) that will be used for error determination; prediction target.
        @param write  if set to false, no data is written to SDF. This is used internally to test errors without tainting SDF

        @return  An array with errors in the format of [index,error,c0,c1,cn....] if err is defined, otherwise True
        """
        for i in range(steps):
            try:
                self.loadVariables(self.dictSlice(self.index))
                self.Calculate()
            except OverflowError:
                print("Overflow for i:{}-{}".format(self.index,self.constants))
                self.resetIndex()

            if write:
                self.writeSlice()
            self.stepIndex()
            if err is not None:
                try:
                    self.error.append([self.index,
                                       abs(self.dictSlice(self.index)[err] - self.predictors[err]),
                                       self.dictSlice(self.index)[err],
                                       self.predictors[err],
                                       *self.dictToArray(self.constants)])
                except OverflowError:
                    self.error.append([self.index,10**9,*self.dictToArray(self.constants)])
        if err is not None:
            return self.error
        else:
            return True

    def SimSolve(self,steps,repeats,err):
        """! Subsequent simulating and evolution based algorithm to test results

        @param steps  amount of iteration to optimize each parameter that is in the solveList and steps to simulate
        @param repeats  amount of times to repeat this loop of simulate -> evolve
        @param err  name of predictor (and dataframe column) that will be used for error determination; prediction target.

        @return  An array with errors in the format of [index,error,c0,c1,cn....]
        """
        for i in range(repeats):
            self.Simulate(steps,err)
            self.Evolver(err,steps)
        return self.error


def main():
    """! Example main to run if file is not being imported
    """
    a = MultiFitter("testInstance",verbose=False)
    a.loadData(SimThis)
    a.loadConstants({"C":120.0,"U":1.0})

    df = pd.DataFrame(a.error,columns=["index","error","c0","c1"])
    a.dt = 60.0
    pd.DataFrame(a.SimSolve(2,2,"Tobj"),columns=["index","error","y","ypred","c0","c1"]).plot()

if __name__ == "__main__":
    print("Mufit is being run directly")
    main()
else:
    print("Mufit is being imported")
