# Fast Iterative Multi Fitter framework
This package was written with the intention to speed up the process of fitting models to data. The mathematical model is
expressed as a set of lambda functions which can have circular references to itself and other components. Pandas
dataframes are used as an internal format and for mapping data to the lambdafunctions.
A set of convenience functions is supplied to ease loading data, loading coefficients, plotting errors and more.

Below an example lambdadict and lambdamap are shown.
* In Lambdamap $ is used to assign coefficients, these will be modified with the fitting methods
* Constants are indicated with a \* and are not modified.

```
     lambdaMap = {"Qtr": ["Tobj","Tenv","$U"],
                  "Qsol":["Psol"],
                  "Qel":["Pe"],
                  "dTobj":["Qtr","Qsol","Qel","dt","$C"],
                  "Tobj":["Tobj","dTobj"]}

     lambdaDict = {"Qtr": lambda Tobj, Tenv, U: (Tenv-Tobj)*0.001/ U,
                   "Qsol": lambda Psol: Psol*0.001,
                   "Qel": lambda Pe: Pe*0.001,
                   "dTobj": lambda Qtr, Qsol, Qel, dt, C: (Qtr+Qsol+Qel)*dt/C,
                   "Tobj": lambda Tobj, dTobj: Tobj + dTobj}
```

## Instantiate the class
 * prototype: `fmf = MultiFitter(lambdaMap=None,lambdaDict=None,verbose=0)`

## Load Data
 * Profilegenerator is included and generates some dummy data `SimThis`
 * load data with `fmf.loadData(SimThis,initPredictors=True)`
 * loadData expects a pandas dataframe with column names as defined in lambdaMap

## Possibly provide some constraints and run the model on the data
 * A dictionary with "name":[min,max]" values can be supplied to limit the search space
 * `SMap = {"U":[0.01,25.0],"C":[10,2000]}`
 * `error = fmf.BulkError([25.0,300],"Tobj")`
 * `errors = fmf.BulkEvolver("Tobj",constraints=SMap)`
 * `errorsscipy = fmf.scipyOptimize("Tobj",constraints=SMap)`

The BulkError method will return the accumulated error for the provided coefficients, where Tobj is the target
predictor, for error calculation. The BulkEvolver method will return an array of arrays sorted from best to worst:
[[error,c0,c1,c2,cn]] where c0,c1,cn are the coefficients in order.
Errorsscipy will return the result of nelder-mead scipy solver. Constraints are optional.

## Analyze results
 * All mentioned methods already provide some form of error-coefficient feedback to make quick decisions.
 * with `fmf.plotError(All=True)` a 3-dimensional chart will be given for each parameter with index,error,coefficient as
   axis.
 * with `fmf.loadConstants(constDict=dict)` a dictionary with "name":value pairs will be set for internal coefficients.
 * with `fmf.ArrayToConstants(array)` an array with ordered coefficients will be applied for future calculations.
 * with `fmf.resetIndex()` the index is reset to the beginning of the dataset
 * with `Simulation = fmf.simulate(fmf.dataLength,"Tobj")[-fmf.datalength:]` a fresh simulation will be stored, note
   that this returns an array with [index,error,y,y-predicted,c0,c1,c2,cn...]
 * To plot this nicely plotly is suitable; make sure that the pandas plotting backend is set to plotly
   * `pd.DataFrame(Simulation,columns=fmf.columns).plot()
