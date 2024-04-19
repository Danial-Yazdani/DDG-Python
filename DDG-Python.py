import numpy as np
import math

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
        self.CurrentBestSolution = np.nan * np.ones(self.NumberOfVariables)
        self.CurrentBestSolutionValue = np.inf

        # Defining dataset and sampling parameters
        self.data = {
            'dataset': np.empty((0, self.NumberOfVariables)),  # Initialize an empty dataset
            'size': 1000,  # Maximum size of the dataset
            'FrequentSamplingLikelihood': 0.01, # The likelihood of Incremental Sampling
            'IncrementalSamplingSize': math.ceil(self.SampleSize * 0.05)  # Define the percentage of dataset to be replaced by new samples
        }
        self.data_generation(self.data['size'])


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
            
        dgc.WeightSeverity = self.LocalWeightSeverityRange[0] + ((self.LocalWeightSeverityRange[1] - self.LocalWeightSeverityRange[0]) * self.rng.random())
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
    
        # Generate a new sample based on the DGCs
    def data_generation(self, new_sample_size):
        data_sample = np.full((new_sample_size, self.NumberOfVariables), np.nan)
        weights = np.array([dgc.weight for dgc in self.dgc])
        probability = weights / weights.sum()  # Probability of selecting each DGC

        counter = 0
        while counter < new_sample_size:
            chosen_id = self.rng.choice(len(self.dgc), p=probability)
            random_vector = self.rng.standard_normal(self.NumberOfVariables)
            sample = (random_vector * self.dgc[chosen_id].sigma @ self.dgc[chosen_id].RotationMatrix) + self.dgc[chosen_id].center

            # Check if the sample needs to be within certain coordinates (uncomment if needed)
            # if np.all(sample >= self.min_coordinate) and np.all(sample <= self.max_coordinate):
            data_sample[counter, :] = sample
            counter += 1

        # Add new samples to the beginning of the dataset and manage the dataset size
        self.data['dataset'] = np.vstack([data_sample, self.data['dataset']])
        self.data['dataset'] = self.data['dataset'][:self.data['size'], :]  # Truncate to maintain size


    #Evaluating clustering solutions. This objective function calculates the sum of intra-cluster distances for each solution.
    def clustering_evaluation(self, X):
        if len(X.shape)<2:
            X = X.reshape(1,-1)        
        SolutionNumber = X.shape[0]
        result = np.full(SolutionNumber, np.nan)
        for i in range(SolutionNumber):
            if self.FE > self.MaxEvals:
                return result  # Termination criterion has been met
            x = X[i, :]
            cluster_center_position = x.reshape(self.NumberOfVariables, self.cluster_number).T
            distances = np.linalg.norm(self.data['dataset'][:, None, :] - cluster_center_position[None, :, :], axis=2)
            closest_cluster_indices = np.argmin(distances, axis=1)
            selected_distances = distances[np.arange(distances.shape[0]), closest_cluster_indices]
            result[i] = np.sum(selected_distances)  # Sum of intra-cluster distances
            self.FE += 1
            # For performance measurement
            if result[i] < self.CurrentBestSolutionValue:  # for minimization
                self.CurrentBestSolution = x
                self.CurrentBestSolutionValue = result[i]
            self.BestValueAtEachFE[self.FE] = self.CurrentBestSolutionValue
            # changes in the landscape and dataset
            self.check_for_environmental_changes()
        return result
    

    #Check for environmental changes and trigger the environmental change generator
    def check_for_environmental_changes(self):
        recent_large_change_flag = False
        dgc_index = 0
        for dgc in self.dgc:
            if self.rng.random() < dgc.LocalChangeLikelihood:
                self.environmental_change_generator(dgc_index) # local change for DGC jj, the change code is a non-negative integer
            dgc_index += 1 
        if self.rng.random() < self.GlobalChangeLikelihood:
            self.environmental_change_generator(-1) # -1 is the change code for the global severe changes in DGCs' parameters
            recent_large_change_flag = True
        if self.rng.random() < self.DGCNumberChangeLikelihood:
            self.environmental_change_generator(-2) # -2 is the change code for the change in the number of DGCs
            recent_large_change_flag = True
        if self.rng.random() < self.VariableNumberChangeLikelihood:
            self.environmental_change_generator(-3) # -3 is the change code for the change in the number of variables
            recent_large_change_flag = True
        if self.rng.random() < self.ClusterNumberChangeLikelihood:
            self.environmental_change_generator(-4) # -4 is the change code for the change in the number of clusters
            recent_large_change_flag = True
        # Sampling (updating dataset after changes)
        if recent_large_change_flag:  # Sample all dataset from the updated DGCs
            self.data_generation(self.data['size'])
            self.CurrentBestSolutionValue = self.current_solution_evaluation(self.CurrentBestSolution)
        if self.rng.random() < self.data['FrequentSamplingLikelihood']:  # Incremental sampling based on the fixed frequency
            self.data_generation(self.data['IncrementalSamplingSize'])
            self.CurrentBestSolutionValue = self.current_solution_evaluation(self.CurrentBestSolution)
    
    #The following function evaluates a single clustering solution, employing the same objective function and representation as clustering_evaluation(.).
    #Its primary use is to re-evaluate the current best solution following updates to the dataset, specifically for performance measurement purposes.
    def current_solution_evaluation(self, x):
        cluster_center_position = x.reshape(self.NumberOfVariables, self.ClusterNumber).T
        distances = np.linalg.norm(self.data['dataset'][:, None, :] - cluster_center_position[None, :, :], axis=2)
        closest_cluster_indices = np.argmin(distances, axis=1)
        selected_distances = distances[np.arange(distances.shape[0]), closest_cluster_indices]
        result = np.sum(selected_distances)  # Sum of intra-cluster distances
        return result
    
    # environmental_change_generator updates the parameters of the Dynamic Gaussian Components (DGCs) to simulate environmental changes within the distributions.
    # The parameter 'ChangeCode' specifies the type of change being simulated.
    def environmental_change_generator(self, change_code):
        # Local change for a specific DGC (positive integer change codes)
        if change_code >= 0:
            dgc = self.dgc[change_code]
            # Update DGC center
            random_direction = self.rng.standard_normal(self.NumberOfVariables)
            random_direction /= np.linalg.norm(random_direction)  # Normalize to unit vector
            summed_vector = ((1 - dgc.ShiftCorrelationFactor) * random_direction) + \
                            (dgc.ShiftCorrelationFactor * dgc.PreviousShiftDirection)
            relocation_direction = summed_vector / np.linalg.norm(summed_vector)
            update_amount = abs(self.rng.standard_normal()) * dgc.ShiftSeverity
            dgc.center += relocation_direction * update_amount
            # Ensure center remains within bounds
            dgc.center = np.clip(dgc.center, self.min_coordinate, self.max_coordinate)
            dgc.PreviousShiftDirection = relocation_direction

            # Update weight
            if self.rng.random() < dgc.DirectionChangeProbability:
                dgc.WeightDirection *= -1
            dgc.weight += dgc.WeightSeverity * dgc.WeightDirection
            dgc.weight = max(min(dgc.weight, self.max_weight), self.min_weight)

            # Update sigma
            if self.rng.random() < dgc.DirectionChangeProbability:
                dgc.SigmaDirection *= -1
            dgc.sigma += dgc.SigmaSeverity * dgc.SigmaDirection
            dgc.sigma = np.clip(dgc.sigma, self.min_sigma, self.max_sigma)

            # Update rotation if applicable
            if self.rotation == 1:
                theta_changes = self.rng.standard_normal((self.NumberOfVariables, self.NumberOfVariables))
                theta_changes = np.triu(theta_changes, 1)  # Only upper triangle affects the rotation
                dgc.ThetaMatrix += theta_changes * dgc.RotationSeverity * dgc.RotationDirection
                dgc.ThetaMatrix = np.clip(dgc.ThetaMatrix, self.MinAngle, self.MaxAngle)
                dgc.RotationMatrix = self.GenerateRotationMatrix(dgc.ThetaMatrix)

        elif change_code == -1:  # Global severe changes
            for dgc in self.dgc:
                self.apply_global_changes(dgc)

        elif change_code == -2:  # Change in the number of DGCs
            self.adjust_dgc_count()

        elif change_code == -3:  # Change in the number of variables
            self.adjust_variable_count()

        elif change_code == -4:  # Change in the number of clusters
            self.adjust_cluster_count()

    def apply_global_changes(self, dgc):
        # Similar logic to local changes but applies globally with potentially higher severity
        # Example for center update; apply similar logic for weight, sigma, and rotation if applicable
        random_direction = self.rng.standard_normal(self.NumberOfVariables)
        random_direction /= np.linalg.norm(random_direction)
        update_amount = abs(self.rng.standard_normal()) * self.GlobalShiftSeverityValue
        dgc.center += random_direction * update_amount
        dgc.center = np.clip(dgc.center, self.min_coordinate, self.max_coordinate)

    def adjust_dgc_count(self):
        current_count = len(self.dgc)
        change = self.rng.integers(-1, 2)  # Randomly decide to add or remove one DGC
        new_count = current_count + change
        new_count = max(min(new_count, self.MaxDGCnumber), self.MinDGCnumber)
        if new_count < current_count:
            self.dgc = self.dgc[:new_count]
        else:
            for _ in range(new_count - current_count):
                new_dgc = self.DGC()  # Assuming a method to initialize a new DGC properly
                self.initialize_dgc(new_dgc)  # Assuming a method to initialize DGC properties
                self.dgc.append(new_dgc)

    def adjust_variable_count(self):
        # This would adjust the number of variables in each DGC; details depend on your model's specifics
        pass

    def adjust_cluster_count(self):
        # This would adjust the number of cluster centers in the clustering algorithm; specifics may vary
        pass


    

ddg = DDG()