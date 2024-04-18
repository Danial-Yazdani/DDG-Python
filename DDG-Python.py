import numpy as np

class DDG:
    # Define the DGC class that contains the parameters of each Dynamic Gaussian Component
    class DGC:
        def __init__(self):
            self.center = None
            self.weight = None
            self.sigma = None
            self.RotationMatrix = None
            self.ThetaMatrix = None
            self.ShiftSeverity = None
            self.ShiftCorrelationFactor = None
            self.PreviousShiftDirection = None
            self.SigmaSeverity = None
            self.SigmaDirection = None
            self.WeightSeverity = None
            self.WeightDirection = None
            self.RotationSeverity = None
            self.RotationDirection = None
            self.LocalChangeLikelihood = None
            self.DirectionChangeProbability = None

    def __init__(self):
        # Initialize the parameters of DDG

        # Seed for random number generation used in DDG
        self.seed = 2151
        # Create a RandomState instance with a specific seed for DDG
        self.rng = np.random.default_rng(self.seed)

        self.MaxEvals = 500000  # Maximum function evaluation number. Used for terminating the simulation.

        # Number of DGCs, variables, and clusters
        self.MinNumberOfVariables = 2
        self.MaxNumberOfVariables = 5
        self.NumberOfVariables = 5  # Set it to a specific initial value, or comment it and use the following line to randomly initialize it
        # Uncomment the next line if you want to randomly initialize the number of variables
        # self.NumberOfVariables = self.rng.integers(self.MinNumberOfVariables, self.MaxNumberOfVariables + 1)

        self.min_cluster_number = 2
        self.max_cluster_number = 10
        self.cluster_number = 5  # Set it to a specific initial value, or comment it and use the following line to randomly initialize it
        # Uncomment the next line if you want to randomly initialize the number of clusters
        # self.cluster_number = self.rng.integers(self.min_cluster_number, self.max_cluster_number + 1)

        self.MinDGCnumber = 3
        self.MaxDGCnumber = 10
        self.DGCnumber = 7  # Set it to a specific initial value, or comment it and use the following line to randomly initialize it
        # Uncomment the next line if you want to randomly initialize the number of DGCs
        # self.DGCnumber = self.rng.integers(self.MinDGCnumber, self.MaxDGCnumber + 1)

        # Defining DGCs
        self.dgc = [self.DGC() for _ in range(self.DGCnumber)]

        # Initializing the center positions of DGCs
        self.min_coordinate = -70  # Used for bounding the center (mean) positions of DGCs
        self.max_coordinate = 70  # Used for bounding the center (mean) positions of DGCs
        for dgc in self.dgc:
            # Initialize the mean position (center) of each DGC randomly in the specified range
            self.initialize_DGC_center(dgc)

            
        # Defining the weight values of DGCs
        self.min_weight = 1
        self.max_weight = 3
        for dgc in self.dgc:
            # Initialize the weight of each DGC randomly in the specified range
            self.initialize_DGC_weights(dgc)


        # Defining the sigma values of DGCs
        self.min_sigma = 7
        self.max_sigma = 20
        self.Conditioning = 1  # (0) Condition number is 1 for all DGCs but the sigma values are different from a DGC to another.
                               # (1) Condition number is random for all DGCs.
        for dgc in self.dgc:
            # Initialize the sigma values of each DGC randomly in the specified range and based on the conditioning type
            self.initialize_DGC_sigmas(dgc)
            
            
        # Defining the rotation matrices of DGCs
        self.MinAngle = -np.pi
        self.MaxAngle = np.pi
        self.rotation = 1 # (0) Without rotation
                          # (1) Random Rotation for all DGCs==> Rotation with random angles for each plane for each DGC    
        for dgc in self.dgc:            
            # Initialize the rotation matrices of DGCs based on the rotation type, and where the rotation angles are randomly chosen in the specified range
            self.initialize_rotations(dgc)
        

        # Set severity values for Gradual local changes for each DGC
        # For parameters that are not going to be impacted in environmental changes (i.e., remain fixed over time), set the severity range to [0,0].
        self.LocalShiftSeverityRange = [0.1, 0.2]
        self.RelocationCorrelationRange = [0.99, 0.995]
        self.LocalSigmaSeverityRange = [0.05, 0.1]
        self.LocalWeightSeverityRange = [0.02, 0.05]
        self.LocalRotationSeverityRange = [np.pi / 360, np.pi / 180]
        self.DirectionChangeProbabilityRange = [0.02, 0.05]
        self.LocalTemporalSeverityRange = [0.05, 0.1]
        for dgc in self.dgc:  
            # Initialize the severity values for Gradual local changes for each DGC based on the specified ranges          
            self.initialize_severity(dgc)

        # Change severity values for severe changes in the parameters of all DGCs
        self.GlobalShiftSeverityValue = 10
        self.GlobalSigmaSeverityValue = 5
        self.GlobalWeightSeverityValue = 0.5
        self.GlobalAngleSeverityValue = np.pi / 4
        self.GlobalSeverityControl = 0.1
        self.GlobalChangeLikelihood = 0.0001

        # Parameters for changing the numbers of variables, DGCs, and cluster centers
        self.DGCNumberChangeSeverity = 1
        self.VariableNumberChangeSeverity = 1
        self.ClusterNumberChangeSeverity = 1
        self.DGCNumberChangeLikelihood = 0.0001
        self.VariableNumberChangeLikelihood = 0.0001
        self.ClusterNumberChangeLikelihood = 0.0001

        # Parameters used for storing the results
        self.BestValueAtEachFE = np.inf * np.ones(self.MaxEvals)
        self.FE = 0
        self.CurrentBestSolution = None
        self.CurrentBestSolutionValue = np.inf

 
    # Initialize the mean (center) position of a DGC
    def initialize_DGC_center(self, dgc):
        dgc.center = self.min_coordinate + (self.max_coordinate - self.min_coordinate) * self.rng.random(self.NumberOfVariables)

    # Initialize the weight of a DGC
    def initialize_DGC_weights(self, dgc):
        dgc.weight = self.min_weight + (self.max_weight - self.min_weight) * self.rng.random()


    # Initialize the sigma values of a DGC
    def initialize_DGC_sigmas(self, dgc):
        if self.Conditioning == 0:
            dgc.sigma = (self.min_sigma + (self.max_sigma - self.min_sigma) * self.rng.random()) * np.ones(self.NumberOfVariables)
        elif self.Conditioning == 1:
            dgc.sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * self.rng.random(self.NumberOfVariables)
        else:
            print('Warning: Wrong number is chosen for conditioning.')

    # Initialize the rotation matrices of DGCs
    def initialize_rotations(self, dgc):
        if self.rotation == 1:
            ThetaMatrix = np.zeros((self.NumberOfVariables, self.NumberOfVariables))
            upper_triangle = np.triu_indices(self.NumberOfVariables, 1)
            ThetaMatrix[upper_triangle] = self.MinAngle + (self.MaxAngle - self.MinAngle) * self.rng.random(len(upper_triangle[0]))
            dgc.ThetaMatrix = ThetaMatrix
            dgc.RotationMatrix = self.GenerateRotationMatrix(ThetaMatrix)
        else:
            dgc.RotationMatrix = np.eye(self.NumberOfVariables)
            dgc.ThetaMatrix = np.zeros((self.NumberOfVariables, self.NumberOfVariables))


    # Initialize the severity values for Gradual local changes for each DGC        
    def initialize_severity(self, dgc):
        dgc.ShiftSeverity = self.LocalShiftSeverityRange[0] + ((self.LocalShiftSeverityRange[1] - self.LocalShiftSeverityRange[0]) * self.rng.random())
        dgc.ShiftCorrelationFactor = self.RelocationCorrelationRange[0] + ((self.RelocationCorrelationRange[1] - self.RelocationCorrelationRange[0]) * self.rng.random())
        tmp = self.rng.standard_normal(self.NumberOfVariables)
        dgc.PreviousShiftDirection = tmp / np.sqrt(np.sum(tmp ** 2))

        dgc.SigmaSeverity = self.LocalSigmaSeverityRange[0] + ((self.LocalSigmaSeverityRange[1] - self.LocalSigmaSeverityRange[0]) * self.rng.random())
        if self.Conditioning == 0:
            dgc.SigmaDirection = np.ones(self.NumberOfVariables) * (self.rng.integers(2) * 2 - 1)
        else:
            dgc.SigmaDirection = self.rng.integers(2, size=self.NumberOfVariables) * 2 - 1
            
        dgc.WeightSeverity = self.LocalWeightSeverityRange[0] + ((self.LocalWeightSeverityRange[1] - self.LocalWeightSeverityRange[0]) * self.rng.uniform())
        dgc.WeightDirection = self.rng.integers(2) * 2 - 1
            
        dgc.RotationSeverity = self.LocalRotationSeverityRange[0] + ((self.LocalRotationSeverityRange[1] - self.LocalRotationSeverityRange[0]) * self.rng.random())
        dgc.RotationDirection = np.triu(self.rng.integers(2, size=(self.NumberOfVariables, self.NumberOfVariables)) * 2 - 1, 1)

        dgc.LocalChangeLikelihood = self.LocalTemporalSeverityRange[0] + ((self.LocalTemporalSeverityRange[1] - self.LocalTemporalSeverityRange[0]) * self.rng.random())
        dgc.DirectionChangeProbability = self.DirectionChangeProbabilityRange[0] + ((self.DirectionChangeProbabilityRange[1] - self.DirectionChangeProbabilityRange[0]) * self.rng.random())


    # Generate a rotation matrix based on the Theta matrix for a DGC
    def GenerateRotationMatrix(self, theta):
        R = np.eye(self.NumberOfVariables)
        for p in range(self.NumberOfVariables - 1):
            for q in range(p + 1, self.NumberOfVariables):
                if theta[p, q] != 0:
                    G = np.eye(self.NumberOfVariables)
                    cos_val = np.cos(theta[p, q])
                    sin_val = np.sin(theta[p, q])
                    G[p, p], G[q, q] = cos_val, cos_val
                    G[p, q], G[q, p] = -sin_val, sin_val
                    R = np.dot(R, G)
        return R



ddg = DDG()