from utils.glmneuron import *
from utils.BasePostProcessor import *


def get_rough_penalty(param,modelType, numPos=400, numHD=18, numSpd=10, numTheta=18):
    # roughness regularizer weight - note: these are tuned using the sum of f,
    # and thus have decreasing influence with increasing amounts of data
    b_pos = 5e0
    b_hd = 5e1
    b_spd = 5e2 #5e1
    b_th = 5e1

     # initialize parameter-relevant variables
    J_pos = 0
    J_pos_g = np.array([])
    J_pos_h = None

    J_hd = 0
    J_hd_g = np.array([])
    J_hd_h = None

    J_spd = 0
    J_spd_g = np.array([])
    J_spd_h = None

    J_theta = 0
    J_theta_g = np.array([])
    J_theta_h = None

    # find parameters
    (param_pos, param_hd, param_spd, param_theta) = find_param(param, modelType,
                                                               numPos, numHD, numSpd, numTheta)

    # Compute the contribution for f, df, and the hessian
    if param_pos is not None:
        (J_pos, J_pos_g, J_pos_h) = rough_penalty_2d(param_pos, b_pos)

    if param_hd is not None:
        (J_hd, J_hd_g, J_hd_h) = rough_penalty_1d_circ(param_hd, b_hd)

    if param_spd is not None:
        (J_spd, J_spd_g, J_spd_h) = rough_penalty_1d(param_spd, b_spd)

    if param_theta is not None:
        (J_theta, J_theta_g, J_theta_h) = rough_penalty_1d_circ(
            param_theta, b_th)

    
    return {'J':(J_pos, J_hd, J_spd, J_theta), 'J_g':(J_pos_g, J_hd_g, J_spd_g, J_theta_g), 'J_h': (J_pos_h, J_hd_h, J_spd_h, J_theta_h)}


def ln_poisson_model_jac(param, X, Y, modelType=[1,1,1,0]):
    (J_pos_g, J_hd_g, J_spd_g, J_theta_g) = get_rough_penalty(param,modelType)['J_g']

    u = X*np.matrix(param).T
    rate = np.exp(u)
    J = np.hstack([J_pos_g, J_hd_g, J_spd_g, J_theta_g])
    df = np.real(X.T * (rate - Y) + J[:,None])/Y.shape[0]
    return np.array(df).flatten()


def ln_poisson_model_hessian(param, X, Y,modelType=[1,1,1,0]):

    (J_pos_h, J_hd_h, J_spd_h, J_theta_h) = get_rough_penalty(param,modelType)['J_h']

    u = X*np.matrix(param).T
    rate = np.exp(u)
    rX = np.multiply(rate, X)
    diag = [J_pos_h, J_hd_h, J_spd_h, J_theta_h]
    diag = [d for d in diag if d is not None]
    hessian_glm = (rX.T*X + block_diag(*diag))/X.shape[0]
    return hessian_glm



def rough_penalty_2d(param, beta):
    numParam = (param.ravel().shape[0])

    # make diagnoal matrix
    n = np.sqrt(numParam).astype(int)
    D1 = np.ones((n, 1))*np.array([-1, 1])
    D1 = np.diag(D1[:, 1])+np.diag(D1[1:, 0], -1)
    D1 = D1[1:, :]
    DD1 = D1.T@D1

    M1 = np.kron(np.eye(n), DD1)
    M2 = np.kron(DD1, np.eye(n))
    M = (M1+M2)
    J = beta*0.5*param.T@M@param
    J_g = beta*M@param.T
    J_h = beta*M
    
    return (J, J_g, J_h)


def ln_poisson_model(param, X, Y, modelType=[1,1,1,0]):
    # X: navigation variables, Y: spiketrain
    u = X*np.matrix(param).T
    rate = np.exp(u)


    # compute f, the gradient, and the hessian
    # print(J_pos)
    # print(J_hd)
    # print(J_spd)
    (J_pos, J_hd, J_spd, J_theta) = get_rough_penalty(param,modelType)['J']
    n = Y.shape[0]
    f = np.mean(rate-np.multiply(Y, u)) + (J_pos + J_hd + J_spd + J_theta)/n
    return f

def rough_penalty_1d_circ(param, beta):
    n = (param.ravel().shape[0])

    # make diagnoal matrix
    D1 = np.ones((n, 1))*np.array([-1, 1])
    D1 = np.diag(D1[:, 1])+np.diag(D1[1:, 0], -1)
    D1 = D1[1:, :]
    DD1 = D1.T@D1

    # correct for smooth in first and last bin
    DD1[0, :] = np.roll(DD1[1, :], -1, axis=0)
    DD1[-1, :] = np.roll(DD1[-2, :], 1, axis=0)

    J = beta*0.5*param.T@DD1@param
    J_g = beta*DD1@param.T
    J_h = beta*DD1

    return (J, J_g, J_h)


def rough_penalty_1d(param, beta):
    n = param.ravel().shape[0]
    D1 = np.ones((n, 1))*np.array([-1, 1])
    D1 = np.diag(D1[:, 1])+np.diag(D1[1:, 0], -1)
    D1 = D1[1:, :]
    DD1 = D1.T@D1

    J = beta*0.5*param.T@DD1@param
    J_g = beta*DD1@param.T
    J_h = beta*DD1

    return (J, J_g, J_h)

def crossValidate(spiketrain, datagrid, modelType, dt, speed=None, valFolds=10):
    # Do cross validation on the data and obtain their performance metrics
    prefList = []
    
    for i in range(valFolds):
        (datagrid_train, datagrid_test) = getCrossValidationFold(datagrid, valFolds, i)
        (spiketrain_train, spiketrain_test) = getCrossValidationFold(spiketrain, valFolds, i)
        model = GLMPostProcessor(spiketrain_train, datagrid_train, modelType, dt, speed)
        model.run()
        performance = model.evalPerformance(spiketrain_test, datagrid_test)
        performance_test = addKeyPrefix(performance, 'test_')
        performance = model.evalPerformance(spiketrain_train, datagrid_train)
        performance_train = addKeyPrefix(performance, 'train_')

        prefList.append({**performance_test, **performance_train})

    # Merge the list into a single dictionary
    prefDict = {k: np.array([x[k] for x in prefList]) for k in prefList[0].keys()}
    
    return prefDict

class GLMPostProcessor(BasePostProcessor):
    def __init__(self,spiketrain, datagrid, modelType, dt, speed=None, speedLimit=50):
        """Initiate a GLM processor
        
        Arguments:
            spiketrain {np.narray} -- spike train in the form of time x cluster
            datagrid {np.narray} -- parameter to fit, in the form of time x bin
            modelType {list} -- a list indicating the model type
        
        Keyword Arguments:
            speed {np.array} -- speed vector used for quality control (default: {None})
            speedLimit {float} -- speed limit above which the data point will be removed (default: {50})
        """
        super().__init__()

        if speed is not None:
            too_fast = np.where(speed >= speedLimit)
            datagrid = np.delete(datagrid, too_fast, axis=0)
            spiketrain = np.delete(spiketrain, too_fast, axis=0)

        self.spiketrain = self.makeMatrix(spiketrain)
        self.dt = dt

        self.modelType = modelType
        self.datagrid = datagrid
        self.result = None

    def makeMatrix(self,x):
        if x.ndim ==1:
            return x[:,None]
        else:
            return x

    def optimize(self, X,Y):
        #Optimize on the model parameters

        numCol = X.shape[1]
        param = np.random.randn(numCol)*1e-3

        return minimize(ln_poisson_model,param,args=(X,Y,self.modelType),
            method='Newton-CG',jac=ln_poisson_model_jac, hess=ln_poisson_model_hessian)


    def run(self):
        """Optimize on the LN poisson model
        
        Returns:
            optimization object -- result from scipy minimize function
        """
        self.result= self.optimize(self.datagrid, self.spiketrain)
        return self.result

    def evalPerformance(self, spike_train, datagrid):
        #calculate the metrics of the fit

        spike_train = self.makeMatrix(spike_train)

        if self.result:
            self.performance = compare_model_performance(self.result.x,datagrid,spike_train,self.dt)
            self.tunningcurve = self.getTuningCurves(spike_train, datagrid, self.dt)
            return {**self.performance, **self.tunningcurve}
        else:
            raise ValueError("Error: no model parameters. Please fit the model first.")
    
    def predict(self,datagrid):
        if self.result:
            param = self.result.x
            return np.exp(datagrid*np.matrix(param).T)
        else:
            raise ValueError("Error: no model parameters. Please fit the model first.")

    def getTuningCurves(self, spiketrain_bin,datagrid, dt):
        # Return the tuning curves
        # some data point may be empty if no spike is found at that spot
        dataIdx = np.argmax(datagrid, axis=1) #convert one-hot to index
        count=np.zeros((datagrid.shape[1],))
        std = np.zeros((datagrid.shape[1],))
        se = np.zeros((datagrid.shape[1],))
        for i in range(count.shape[0]):
            spikeinBin = spiketrain_bin[i==dataIdx]
            if spikeinBin.size>0:
                count[i] = np.mean(spikeinBin)/dt
                std[i] = np.std(spikeinBin/dt)
                se[i] = std[i]/spikeinBin.size

            else:
                count[i]= 0
                std[i] = 0
                se[i] = 0
        
        if self.result:
            model_curve = np.exp(self.result.x)/dt
        else:
            model_curve = None
        
        return {'firing_rate':count, 'firing_rate_std': std, 
            'firing_rate_se':se, 'model_curve': model_curve}

    def cleanup(self):
        pickle.dump(self.result,open('GLM_tuning.pkl','wb'))