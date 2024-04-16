import numpy as np

class DDG:
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
        self.seed = 2151
        # Create a RandomState instance with a specific seed for DDG
        self.rng = np.random.default_rng(self.seed)

        self.max_evals = 500000  # Maximum function evaluation number

        # Number of DGCs, variables, and clusters
        self.MinNumberOfVariables = 2
        self.MaxNumberOfVariables = 5
        self.NumberOfVariables = 2  # Static initialization
        # Uncomment the next line if you want to randomly initialize the number of variables
        # self.NumberOfVariables = self.rng.integers(self.MinNumberOfVariables, self.MaxNumberOfVariables + 1)

        self.min_dgc_number = 3
        self.max_dgc_number = 10
        self.dgc_number = 7  # Static initialization
        # Uncomment the next line if you want to randomly initialize the number of DGCs
        # self.dgc_number = self.rng.integers(self.min_dgc_number, self.max_dgc_number + 1)

        self.min_cluster_number = 2
        self.max_cluster_number = 10
        self.cluster_number = 5  # Static initialization
        # Uncomment the next line if you want to randomly initialize the number of clusters
        # self.cluster_number = self.rng.integers(self.min_cluster_number, self.max_cluster_number + 1)

        # Initializing the center positions of DGCs
        self.min_coordinate = -70  # Used for bounding the center (mean) positions of DGCs
        self.max_coordinate = 70  # Used for bounding the center (mean) positions of DGCs
        self.dgc = [self.DGC() for _ in range(self.dgc_number)]

        self.initialize_dgc_centers()

        # Defining the weight values of the DGCs
        self.min_weight = 1
        self.max_weight = 3
        self.initialize_dgc_weights()

        self.initialize_rotations()


        # Set severity values for Gradual local changes for each DGC
        # For parameters that are not going to be impacted in environmental changes (i.e., remain fixed over time), set the severity range to [0,0].
        self.LocalShiftSeverityRange = [0.1, 0.2]
        self.RelocationCorrelationRange = [0.99, 0.995]
        self.LocalSigmaSeverityRange = [0.05, 0.1]
        self.LocalWeightSeverityRange = [0.02, 0.05]
        self.LocalRotationSeverityRange = [np.pi / 360, np.pi / 180]
        self.DirectionChangeProbabilityRange = [0.02, 0.05]
        self.LocalTemporalSeverityRange = [0.05, 0.1]
        self.initialize_severity()

        self.GlobalShiftSeverityValue = 10
        self.GlobalSigmaSeverityValue = 5
        self.GlobalWeightSeverityValue = 0.5
        self.GlobalAngleSeverityValue = np.pi / 4
        self.GlobalSeverityControl = 0.1
        self.GlobalChangeLikelihood = 0.0001

        self.DGCNumberChangeSeverity = 1
        self.VariableNumberChangeSeverity = 1
        self.ClusterNumberChangeSeverity = 1
        self.DGCNumberChangeLikelihood = 0.0001
        self.VariableNumberChangeLikelihood = 0.0001
        self.ClusterNumberChangeLikelihood = 0.0001

        self.BestValueAtEachFE = np.inf * np.ones(self.MaxEvals)
        self.FE = 0
        self.CurrentBestSolution = None
        self.CurrentBestSolutionValue = np.inf

    def initialize_severity(self):
        for dgc in self.dgc:
            dgc.ShiftSeverity = self.LocalShiftSeverityRange[0] + ((self.LocalShiftSeverityRange[1] - self.LocalShiftSeverityRange[0]) * self.rng.random())
            dgc.ShiftCorrelationFactor = self.RelocationCorrelationRange[0] + ((self.RelocationCorrelationRange[1] - self.RelocationCorrelationRange[0]) * self.rng.random())
            tmp = self.rng.randn(self.NumberOfVariables)
            dgc.PreviousShiftDirection = tmp / np.sqrt(np.sum(tmp ** 2))

            dgc.SigmaSeverity = self.LocalSigmaSeverityRange[0] + ((self.LocalSigmaSeverityRange[1] - self.LocalSigmaSeverityRange[0]) * self.rng.random())
            if self.Conditioning == 0:
                dgc.SigmaDirection = np.ones(self.NumberOfVariables) * (self.rng.randint(2) * 2 - 1)
            else:
                dgc.SigmaDirection = self.rng.randint(2, size=self.NumberOfVariables) * 2 - 1

            dgc.WeightSeverity = self.LocalWeightSeverityRange[0] + ((self.LocalWeightSeverityRange[1] - self.LocalWeightSeverityRange[0]) * self.rng.random())
            dgc.WeightDirection = self.rng.randint(2) * 2 - 1

            dgc.RotationSeverity = self.LocalRotationSeverityRange[0] + ((self.LocalRotationSeverityRange[1] - self.LocalRotationSeverityRange[0]) * self.rng.random())
            dgc.RotationDirection = np.triu(self.rng.randint(2, size=(self.NumberOfVariables, self.NumberOfVariables)) * 2 - 1, 1)

            dgc.LocalChangeLikelihood = self.LocalTemporalSeverityRange[0] + ((self.LocalTemporalSeverityRange[1] - self.LocalTemporalSeverityRange[0]) * self.rng.random())
            dgc.DirectionChangeProbability = self.DirectionChangeProbabilityRange[0] + ((self.DirectionChangeProbabilityRange[1] - self.DirectionChangeProbabilityRange[0]) * self.rng.random())

    def initialize_rotations(self):
        MinAngle = -np.pi
        MaxAngle = np.pi
        rotation = 1 # (0) Without rotation
                     # (1) Random Rotation for all DGCs==> Rotation with random angles for each plane for each DGC        
        for dgc in self.dgc:
            if rotation == 1:
                ThetaMatrix = np.zeros((self.NumberOfVariables, self.NumberOfVariables))
                upper_triangle = np.triu_indices(self.NumberOfVariables, 1)
                ThetaMatrix[upper_triangle] = MinAngle + (MaxAngle - MinAngle) * self.rng.random(len(upper_triangle[0]))
                dgc.ThetaMatrix = ThetaMatrix
                dgc.RotationMatrix = self.rotation(ThetaMatrix)
            else:
                dgc.RotationMatrix = np.eye(self.NumberOfVariables)
                dgc.ThetaMatrix = np.zeros((self.NumberOfVariables, self.NumberOfVariables))



    def initialize_dgc_centers(self):
        for dgc in self.dgc:
            dgc.center = self.min_coordinate + (self.max_coordinate - self.min_coordinate) * self.rng.random(self.NumberOfVariables)

    def initialize_dgc_weights(self):
        for dgc in self.dgc:
            dgc.weight = self.min_weight + (self.max_weight - self.min_weight) * self.rng.random()

    def initialize_dgc_sigmas(self):
        Conditioning = 1  # (0) Condition number is 1 for all DGCs but the sigma values are different from a DGC to another.
                          # (1) Condition number is random for all DGCs.
        if Conditioning == 0:
            for dgc in self.dgc:
                dgc.sigma = (self.min_sigma + (self.max_sigma - self.min_sigma) * self.rng.random()) * np.ones(self.NumberOfVariables)
        elif Conditioning == 1:
            for dgc in self.dgc:
                dgc.sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * self.rng.random(self.NumberOfVariables)
        else:
            print('Warning: Wrong number is chosen for conditioning.')

    def rotation(self, theta):
        R = np.eye(self.Dimension)
        for p in range(self.Dimension - 1):
            for q in range(p + 1, self.Dimension):
                if theta[p, q] != 0:
                    G = np.eye(self.Dimension)
                    cos_val = np.cos(theta[p, q])
                    sin_val = np.sin(theta[p, q])
                    G[p, p], G[q, q] = cos_val, cos_val
                    G[p, q], G[q, p] = -sin_val, sin_val
                    R = np.dot(R, G)
        return R



ddg = DDG()
