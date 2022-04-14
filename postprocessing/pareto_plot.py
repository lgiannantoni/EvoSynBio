import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

import oapackage

#datapoints=np.random.rand(2, 50)
#print(datapoints.shape)

nfiles = 26

for i in range(nfiles):
    fname = str(i)+".tsv"
    datapoints = np.genfromtxt(fname=fname, delimiter="\t", filling_values=0)
    datapoints = datapoints.T
    #print(datapoints.shape)

    #for ii in range(0, datapoints.shape[1]):
    #    w=datapoints[:,ii]
    #    fac=.6+.4*np.linalg.norm(w)
    #    datapoints[:,ii]=(1/fac)*w

    # just to show the data...
    #h=plt.plot(datapoints[1,:], datapoints[2,:], '.b', markersize=12, label='Non Pareto-optimal')
    #_=plt.title('The input data', fontsize=14)
    #plt.xlabel('f0', fontsize=12)
    #plt.ylabel('f1', fontsize=12)
    #plt.xlim([-5, 100])
    #plt.ylim([-5, 100])
    #plt.savefig('ciao.png')



    pareto=oapackage.ParetoDoubleLong()

    for ii in range(0, datapoints.shape[1]):
        w=oapackage.doubleVector( (datapoints[1,ii], datapoints[2,ii]))
        pareto.addvalue(w, ii)

    pareto.show(verbose=1)


    lst=pareto.allindices() # the indices of the Pareto optimal designs

    optimal_datapoints=datapoints[:,lst]

    h=plt.plot(datapoints[1,:], datapoints[2,:], '.b', markersize=12, label='Non Pareto-optimal')
    hp=plt.plot(optimal_datapoints[1,:], optimal_datapoints[2,:], '.r', markersize=12, label='Pareto optimal')
    plt.xlabel('f0', fontsize=12)
    plt.ylabel('f1', fontsize=12)
    #plt.xticks([])
    #plt.yticks([])
    plt.xlim([-5, 100])
    plt.ylim([-5, 100])
    _=plt.legend(loc='upper right', numpoints=1)
    plt.savefig(str(i)+'.png')
    plt.close()



