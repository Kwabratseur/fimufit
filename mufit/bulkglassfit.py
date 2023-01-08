from mufit import *

def main():
    ## Lambdamap, C = constant, other constants are added to solvelist
#     lambdaMap = {"Tavg": ["Tview", "Tenv"],
#                  "Qtr": ["Tobj","Tavg","Pe","$Uh","$Uc"],
#                  "Qrad": ["Tobj","Tview","$Uhr"],
#                  "Qsol":["Psol"],
#                  "Qel":["Pe"],
#                  "dTobj":["Qtr","Qsol","Qel","Qrad","dt","$C"],
#                  "Tobj":["Tobj","dTobj"]}
#     print("Parameter map:{}".format(lambdaMap))
          
#     ## if/else statement in Qtransmission.
#     lambdaDict = {"Tavg": lambda Tview, Tenv: (Tview+Tenv)/2,
#                   "Qtr": lambda Tobj, Tavg, Pe, Uh, Uc: (Tobj-Tavg)*0.001/ Uh if Pe > 0 else (Tobj-Tavg)*0.001/ Uc,
#                   "Qrad": lambda Tobj, Tview, Uhr: (((Tobj+273.15)**4)-((Tview+273.15)**4))*0.001*(5.670374419*10**-8)/Uhr,
#                    "Qsol": lambda Psol: Psol*0.001,
#                    "Qel": lambda Pe: Pe*0.001,
#                    "dTobj": lambda Qtr, Qsol, Qel, Qrad, dt, C: (-Qtr+Qsol+Qel-Qrad)*dt/C,
#                    "Tobj": lambda Tobj, dTobj: Tobj + dTobj}
    
    lambdaMap = {"Tavg": ["Tview", "Tenv"],
             "Qtr": ["Tobj","Tavg","Pe","$Uh","$Uc"],
             "Qrad": ["Tobj","Tview","$Uhr","Pe","$Ucr"],
             "Qsol":["Psol","$Csol"],
             "Qel":["Pe"],
             "dTobj":["Qtr","Qsol","Qel","Qrad","dt","$C"],
             "Tobj":["Tobj","dTobj"]}
    print("Parameter map:{}".format(lambdaMap))

    ## if/else statement in Qtransmission.
    lambdaDict = {"Tavg": lambda Tview, Tenv: (Tview+Tenv)/2,
                  "Qtr": lambda Tobj, Tavg, Pe, Uh, Uc: (Tobj-Tavg)*0.001* Uh if Pe > 0 else (Tobj-Tavg)*0.001* Uc,
                  "Qrad": lambda Tobj, Tview, Uhr, Pe, Ucr: (((Tobj+273.15)**4)-((Tview+273.15)**4))*0.001*(5.670374419*10**-8)*Uhr if Pe > 0 else (((Tobj+273.15)**4)-((Tview+273.15)**4))*0.001*(5.670374419*10**-8)*Ucr,
                   "Qsol": lambda Psol,Csol: Psol*Csol*0.001,
                   "Qel": lambda Pe: Pe*0.001,
                   "dTobj": lambda Qtr, Qsol, Qel, Qrad, dt, C: (-Qtr+Qsol+Qel-Qrad)*dt/C,
                   "Tobj": lambda Tobj, dTobj: Tobj + dTobj}
    
    dateparser = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    droplist = ['   DD','Samples','Lux (Lux)','   FH','   SQ','   RH','    U'] # we don't need rain, wind, wind direction, humidity, sun hours and lux.
    RunData = pd.read_csv("Hoeveler_All_118_519.csv",sep=",", index_col=[0], parse_dates=[0], date_parser=dateparser).drop(droplist,axis=1)#,encoding='ANSI')
    RunData.columns = ['Pt','Tenv','Tview','Pe','Tobj','T23','Te2','Psol']
    Dataset1 = RunData.iloc[58: 1337]
    print("Input dataset:")
    Dataset1.head()

    print("Initializing multifitter class")
    a = MultiFitter(lambdaMap = lambdaMap,lambdaDict=lambdaDict, verbose=2)
    
    print("Load input data and initialize predictors (left side of lambdaMap)")
    a.loadData(Dataset1[:600],initPredictors=True)
    
    print("resample loaded data to 5 minute interval; only works for datetime indexed datasets!")
    a.resampleData(60*5)
    
    print("Plot internal dataset")
    a.DF.plot()
    plt.show()
    
    print("Initializing initial guess for predictors")
    a.loadConstants({"C":120.0,"Uh":36.0,"Uc":1.0,"Uhr":0.4,"Ucr":1.0,"Csol":1.0})
    print(a.parameters())
 
    print("The output format in error plot will be as follows:{}".format(a.columns))
    a.dt = 5*60.0


    print("Below plot shows a simulation/prediction effort for 100 timesteps and error calculated based on 'Tobj'")
    pd.DataFrame(a.Simulate(100,"Tobj"), columns=a.columns).plot()
    plt.show()
    
    SMap = {"Uh":(0.001,45.0),"Uc":(0.0001,45),"Uhr":(0.0001,10),"C":[80,1000],"Ucr":[0.001,25.0],"Csol":[0.001,25.0]}
    
    errs = a.BulkEvolver("Tobj",constraints=SMap)
    for i in errs:
        print(i)
    # print(a.scipyOptimize(err="Tobj"))
    # BestConstants = a.AnalyzeErrors()
    # pd.DataFrame(BestConstants, columns=columns).to_csv("error-constants-scipy.csv")
    # pd.DataFrame(BestConstants, columns=columns).plot()
    # plt.show()

    a.plotError(All=True)
    
    a.SDF.plot()
    plt.show()
    return a