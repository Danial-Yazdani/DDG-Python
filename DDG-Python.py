import numpy as np
import math

class DDG:
    # Define the DGC class that contains the parameters of each Dynamic Gaussian Component
    class DGC:
        def __init__(self):
            self.center = None
            self.weight = None
            self.sigma = None
            self.rotation_matrix = None
            self.theta_matrix = None
            self.shift_severity = None
            self.shift_correlation_factor = None
            self.previous_shift_direction = None
            self.sigma_severity = None
            self.sigma_direction = None
            self.weight_severity = None
            self.weight_direction = None
            self.rotation_severity = None
            self.rotation_direction = None
            self.local_change_likelihood = None
            self.direction_change_probability = None

    def __init__(self):
        # Initialize the parameters of DDG

        # Seed for random number generation used in DDG
        self.seed = 2151
        # Create a RandomState instance with a specific seed for DDG
        self.rng = np.random.default_rng(self.seed)

        self.max_evals = 500000  # Maximum function evaluation number. Used for terminating the simulation.

        # Number of DGCs, variables, and clusters
        self.min_num_of_variables = 2
        self.max_num_of_variables = 5
        self.num_of_variables = 5  # Set it to a specific initial value, or comment it and use the following line to randomly initialize it
        # Uncomment the next line if you want to randomly initialize the number of variables
        # self.num_of_variables = self.rng.integers(self.min_num_of_variables, self.max_num_of_variables + 1)

        self.min_cluster_number = 2
        self.max_cluster_number = 10
        self.cluster_number = 5  # Set it to a specific initial value, or comment it and use the following line to randomly initialize it
        # Uncomment the next line if you want to randomly initialize the number of clusters
        # self.cluster_number = self.rng.integers(self.min_cluster_number, self.max_cluster_number + 1)

        self.min_DGC_number = 3
        self.max_DGC_number = 10
        self.DGC_number = 7  # Set it to a specific initial value, or comment it and use the following line to randomly initialize it
        # Uncomment the next line if you want to randomly initialize the number of DGCs
        # self.DGC_number = self.rng.integers(self.min_DGC_number, self.max_DGC_number + 1)

        # Defining DGCs
        self.dgc = [self.DGC() for _ in range(self.DGC_number)]

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
        self.min_angle = -np.pi
        self.max_angle = np.pi
        self.rotation = 1 # (0) Without rotation
                          # (1) Random Rotation for all DGCs==> Rotation with random angles for each plane for each DGC    
        for dgc in self.dgc:            
            # Initialize the rotation matrices of DGCs based on the rotation type, and where the rotation angles are randomly chosen in the specified range
            self.initialize_rotations(dgc)
        

        # Set severity values for Gradual local changes for each DGC
        # For parameters that are not going to be impacted in environmental changes (i.e., remain fixed over time), set the severity range to [0,0].
        self.local_shift_severity_range = [0.1, 0.2]
        self.relocation_correlation_range = [0.99, 0.995]
        self.local_sigma_severity_range = [0.05, 0.1]
        self.local_weight_severity_range = [0.02, 0.05]
        self.local_rotation_severity_range = [np.pi / 360, np.pi / 180]
        self.direction_change_probability_range = [0.02, 0.05]
        self.local_temporal_severity_range = [0.05, 0.1]
        for dgc in self.dgc:  
            # Initialize the severity values for Gradual local changes for each DGC based on the specified ranges          
            self.initialize_severity(dgc)

        # Change severity values for severe changes in the parameters of all DGCs
        self.global_shift_severity_value = 10
        self.global_sigma_severity_value = 5
        self.global_weight_severity_value = 0.5
        self.global_angle_severity_value = np.pi / 4
        self.global_severity_control = 0.1
        self.global_change_likelihood = 0.0001

        # Parameters for changing the numbers of variables, DGCs, and cluster centers
        self.DGC_number_change_severity = 1
        self.variable_number_change_severity = 1
        self.cluster_number_change_severity = 1
        self.DGC_number_change_likelihood = 0.0001
        self.variable_number_change_likelihood = 0.0001
        self.cluster_number_change_likelihood = 0.0001

        # Parameters used for storing the results
        self.best_value_at_each_FE = np.inf * np.ones(self.max_evals)
        self.FE = 0
        self.current_best_solution = np.nan * np.ones(self.num_of_variables)
        self.current_best_solution_value = np.inf

        # Defining dataset and sampling parameters
        self.data = {
            'dataset': np.empty((0, self.num_of_variables)),  # Initialize an empty dataset
            'size': 1000,  # Maximum size of the dataset
            'FrequentSamplingLikelihood': 0.01, # The likelihood of Incremental Sampling
            'IncrementalSamplingSize': math.ceil(self.data['size'] * 0.05)  # Define the percentage of dataset to be replaced by new samples
        }
        self.data_generation(self.data['size'])


    # Initialize the mean (center) position of a DGC
    def initialize_DGC_center(self, dgc):
        dgc.center = self.min_coordinate + (self.max_coordinate - self.min_coordinate) * self.rng.random(self.num_of_variables)


    # Initialize the weight of a DGC
    def initialize_DGC_weights(self, dgc):
        dgc.weight = self.min_weight + (self.max_weight - self.min_weight) * self.rng.random()


    # Initialize the sigma values of a DGC
    def initialize_DGC_sigmas(self, dgc):
        if self.Conditioning == 0:
            dgc.sigma = (self.min_sigma + (self.max_sigma - self.min_sigma) * self.rng.random()) * np.ones(self.num_of_variables)
        elif self.Conditioning == 1:
            dgc.sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * self.rng.random(self.num_of_variables)
        else:
            print('Warning: Wrong number is chosen for conditioning.')


    # Initialize the rotation matrices of DGCs
    def initialize_rotations(self, dgc):
        if self.rotation == 1:
            theta_matrix = np.zeros((self.num_of_variables, self.num_of_variables))
            upper_triangle = np.triu_indices(self.num_of_variables, 1)
            theta_matrix[upper_triangle] = self.min_angle + (self.max_angle - self.min_angle) * self.rng.random(len(upper_triangle[0]))
            dgc.theta_matrix = theta_matrix
            dgc.rotation_matrix = self.generate_rotation_matrix(theta_matrix)
        else:
            dgc.rotation_matrix = np.eye(self.num_of_variables)
            dgc.theta_matrix = np.zeros((self.num_of_variables, self.num_of_variables))


    # Initialize the severity values for Gradual local changes for each DGC        
    def initialize_severity(self, dgc):
        dgc.shift_severity = self.local_shift_severity_range[0] + \
            ((self.local_shift_severity_range[1] - self.local_shift_severity_range[0]) * self.rng.random())
        dgc.shift_correlation_factor = self.relocation_correlation_range[0] + \
            ((self.relocation_correlation_range[1] - self.relocation_correlation_range[0]) * self.rng.random())
        tmp = self.rng.standard_normal(self.num_of_variables)
        dgc.previous_shift_direction = tmp / np.sqrt(np.sum(tmp ** 2))

        dgc.sigma_severity = self.local_sigma_severity_range[0] + \
            ((self.local_sigma_severity_range[1] - self.local_sigma_severity_range[0]) * self.rng.random())
        if self.Conditioning == 0:
            dgc.sigma_direction = np.ones(self.num_of_variables) * (self.rng.integers(2) * 2 - 1)
        else:
            dgc.sigma_direction = self.rng.integers(2, size=self.num_of_variables) * 2 - 1
            
        dgc.weight_severity = self.local_weight_severity_range[0] + \
            ((self.local_weight_severity_range[1] - self.local_weight_severity_range[0]) * self.rng.random())
        dgc.weight_direction = self.rng.integers(2) * 2 - 1
            
        dgc.rotation_severity = self.local_rotation_severity_range[0] + \
            ((self.local_rotation_severity_range[1] - self.local_rotation_severity_range[0]) * self.rng.random())
        dgc.rotation_direction = np.triu(self.rng.integers(2, size=(self.num_of_variables, self.num_of_variables)) * 2 - 1, 1)

        dgc.local_change_likelihood = self.local_temporal_severity_range[0] + \
            ((self.local_temporal_severity_range[1] - self.local_temporal_severity_range[0]) * self.rng.random())
        dgc.direction_change_probability = self.direction_change_probability_range[0] + \
            ((self.direction_change_probability_range[1] - self.direction_change_probability_range[0]) * self.rng.random())


    # Generate a rotation matrix based on the Theta matrix for a DGC
    def generate_rotation_matrix(self, theta):
        R = np.eye(self.num_of_variables)
        for p in range(self.num_of_variables - 1):
            for q in range(p + 1, self.num_of_variables):
                if theta[p, q] != 0:
                    G = np.eye(self.num_of_variables)
                    cos_val = np.cos(theta[p, q])
                    sin_val = np.sin(theta[p, q])
                    G[p, p], G[q, q] = cos_val, cos_val
                    G[p, q], G[q, p] = -sin_val, sin_val
                    R = np.dot(R, G)
        return R
    
        # Generate a new sample based on the DGCs
    def data_generation(self, new_sample_size):
        data_sample = np.full((new_sample_size, self.num_of_variables), np.nan)
        weights = np.array([dgc.weight for dgc in self.dgc])
        probability = weights / weights.sum()  # Probability of selecting each DGC

        counter = 0
        while counter < new_sample_size:
            chosen_id = self.rng.choice(len(self.dgc), p=probability)
            random_vector = self.rng.standard_normal(self.num_of_variables)
            sample = (random_vector * self.dgc[chosen_id].sigma @ self.dgc[chosen_id].rotation_matrix) + self.dgc[chosen_id].center

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
            if self.FE > self.max_evals:
                return result  # Termination criterion has been met
            x = X[i, :]
            cluster_center_position = x.reshape(self.num_of_variables, self.cluster_number).T
            distances = np.linalg.norm(self.data['dataset'][:, None, :] - cluster_center_position[None, :, :], axis=2)
            closest_cluster_indices = np.argmin(distances, axis=1)
            selected_distances = distances[np.arange(distances.shape[0]), closest_cluster_indices]
            result[i] = np.sum(selected_distances)  # Sum of intra-cluster distances
            self.FE += 1
            # For performance measurement
            if result[i] < self.current_best_solution_value:  # for minimization
                self.current_best_solution = x
                self.current_best_solution_value = result[i]
            self.best_value_at_each_FE[self.FE] = self.current_best_solution_value
            # changes in the landscape and dataset
            self.check_for_environmental_changes()
        return result
    

    #Check for environmental changes and trigger the environmental change generator
    def check_for_environmental_changes(self):
        recent_large_change_flag = False
        dgc_index = 0
        for dgc in self.dgc:
            if self.rng.random() < dgc.local_change_likelihood:
                self.environmental_change_generator(dgc_index) # local change for DGC jj, the change code is a non-negative integer
            dgc_index += 1 
        if self.rng.random() < self.global_change_likelihood:
            self.environmental_change_generator(-1) # -1 is the change code for the global severe changes in DGCs' parameters
            recent_large_change_flag = True
        if self.rng.random() < self.DGC_number_change_likelihood:
            self.environmental_change_generator(-2) # -2 is the change code for the change in the number of DGCs
            recent_large_change_flag = True
        if self.rng.random() < self.variable_number_change_likelihood:
            self.environmental_change_generator(-3) # -3 is the change code for the change in the number of variables
            recent_large_change_flag = True
        if self.rng.random() < self.cluster_number_change_likelihood:
            self.environmental_change_generator(-4) # -4 is the change code for the change in the number of clusters
            recent_large_change_flag = True
        # Sampling (updating dataset after changes)
        if recent_large_change_flag:  # Sample all dataset from the updated DGCs
            self.data_generation(self.data['size'])
            self.current_best_solution_value = self.current_solution_evaluation(self.current_best_solution)
        if self.rng.random() < self.data['FrequentSamplingLikelihood']:  # Incremental sampling based on the fixed frequency
            self.data_generation(self.data['IncrementalSamplingSize'])
            self.current_best_solution_value = self.current_solution_evaluation(self.current_best_solution)
    
    #The following function evaluates a single clustering solution, employing the same objective function and representation as clustering_evaluation(.).
    #Its primary use is to re-evaluate the current best solution following updates to the dataset, specifically for performance measurement purposes.
    def current_solution_evaluation(self, x):
        cluster_center_position = x.reshape(self.num_of_variables, self.ClusterNumber).T
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
            # Update DGC center (mean) positions
            random_direction = self.rng.standard_normal(self.num_of_variables)
            random_direction /= np.linalg.norm(random_direction)  # Normalize to unit vector
            summed_vector = ((1 - dgc.shift_correlation_factor) * random_direction) + \
                            (dgc.shift_correlation_factor * dgc.previous_shift_direction)
            relocation_direction = summed_vector / np.linalg.norm(summed_vector)
            update_amount = abs(self.rng.standard_normal()) * dgc.shift_severity
            UpdatedDGCPosition = dgc.center + (relocation_direction * update_amount)
            UpdatedDGCPosition = np.clip(UpdatedDGCPosition, self.min_coordinate, self.max_coordinate) # Bound the center (mean) position
            relocation_vector = UpdatedDGCPosition - dgc.center
            dgc.previous_shift_direction = relocation_vector / np.linalg.norm(relocation_vector)
            dgc.center = UpdatedDGCPosition
            # Update weights of DGCs
            if self.rng.random() < dgc.direction_change_probability:
                dgc.weight_direction *= -1
            dgc.weight = dgc.weight + (abs(self.rng.standard_normal()) * dgc.weight_severity * dgc.weight_direction)
            dgc.weight = np.clip(dgc.weight, self.min_weight, self.max_weight)

            # Update sigma values of DGCs
            if self.Conditioning == 0: # Keeping the condition number of DGCs as 1
                if self.rng.random() < dgc.direction_change_probability:
                    dgc.sigma_direction *= -1
                dgc.sigma = dgc.sigma + (np.ones(self.num_of_variables)* abs(self.rng.standard_normal()) * dgc.sigma_severity * dgc.sigma_direction)  
            elif self.Conditioning == 1: # Conditioning is not 1 for DGCs
                invert_flags = self.rng.random(self.num_of_variables) < dgc.direction_change_probability
                dgc.sigma_direction[invert_flags] = -dgc.sigma_direction[invert_flags]
                dgc.sigma = dgc.sigma + (abs(self.rng.standard_normal(self.num_of_variables)) * dgc.sigma_severity * dgc.sigma_direction)
            dgc.sigma = np.clip(dgc.sigma, self.min_sigma, self.max_sigma) # Bound the sigma values

            # Update rotation if applicable
            if self.rotation == 1:
                invert_flags = np.triu(self.rng.random((self.num_of_variables, self.num_of_variables)) < dgc.direction_change_probability, 1)
                dgc.rotation_direction[invert_flags] = -dgc.rotation_direction[invert_flags]
                dgc.theta_matrix = dgc.theta_matrix + np.triu(abs(self.rng.standard_normal((self.num_of_variables, self.num_of_variables))), 1) \
                                                * dgc.rotation_severity * dgc.rotation_direction
                dgc.theta_matrix = np.clip(dgc.theta_matrix, self.min_angle, self.max_angle) # Boundary check for angles
                dgc.rotation_matrix = self.generate_rotation_matrix(dgc.theta_matrix)

        elif change_code == -1:  # Severe global change for all DGCs
            for dgc in self.dgc:
                self.apply_global_changes(dgc)

        elif change_code == -2:  # Change in the number of DGCs
            self.adjust_dgc_count()

        elif change_code == -3:  # Change in the number of variables
            self.adjust_variable_count()

        elif change_code == -4:  # Change in the number of clusters
            self.adjust_cluster_count()

    def apply_global_changes(self, dgc):
            random_direction = self.rng.standard_normal(self.num_of_variables)
            random_direction /= np.linalg.norm(random_direction)  # Normalize to unit vector
            dgc.center = dgc.center + (random_direction * self.global_shift_severity_value * \
                                        (2 * self.rng.beta(self.global_severity_control, self.global_severity_control) - 1))
            dgc.center = np.clip(dgc.center, self.min_coordinate, self.max_coordinate)  # Bound the center (mean) position
        
            # Update weights of DGCs
            dgc.weight = dgc.weight + (self.global_weight_severity_value * \
                                          (2 * self.rng.beta(self.global_severity_control, self.global_severity_control) - 1))
            dgc.weight = np.clip(dgc.weight, self.min_weight, self.max_weight) # Bound the weight values

            # Update sigma values of DGCs
            if self.Conditioning == 0: # Keeping the condition number of DGCs as 1
                dgc.sigma = dgc.sigma + ((np.ones(self.num_of_variables) * (2 * self.rng.beta(self.global_severity_control, self.global_severity_control) - 1)) \
                                          * self.global_sigma_severity_value)
            elif self.Conditioning == 1: # Conditioning is not 1 for DGCs
                dgc.sigma = dgc.sigma + ((2 * self.rng.beta(self.global_severity_control, self.global_severity_control, 1, self.num_of_variables) - 1) \
                                         * self.global_sigma_severity_value)
            dgc.sigma = np.clip(dgc.sigma, self.min_sigma, self.max_sigma) # Bound the sigma values

            # Update rotation if applicable
            if self.rotation == 1:
                dgc.theta_matrix = dgc.theta_matrix + (self.global_angle_severity_value * \
                                  np.triu((2 * self.rng.beta(self.global_severity_control, self.global_severity_control, self.num_of_variables, self.num_of_variables) - 1), 1))
                dgc.theta_matrix = np.clip(dgc.theta_matrix, self.min_angle, self.max_angle) # Boundary check for angles
                dgc.rotation_matrix = self.generate_rotation_matrix(dgc.theta_matrix)


    def adjust_dgc_count(self):
        current_count = len(self.dgc)
        new_count = current_count + (self.rng.integers(0, 2) * 2 - 1) * self.DGC_number_change_severity
        new_count = np.clip(new_count, self.min_DGC_number, self.max_DGC_number)
        if new_count < current_count: # Remove DGCs
            number_to_remove = current_count - new_count
            indices_to_remove = set(self.rng.choice(len(self.dgc), number_to_remove, replace=False)) # Randomly select number_to_remove DGCs to remove
            self.dgc = [d for i, d in enumerate(self.dgc) if i not in indices_to_remove] # remove number_to_remove DGCs whose indices are in indices_to_remove
        elif new_count > current_count: # Add new DGCs
            for _ in range(new_count - current_count):
                new_dgc = self.DGC() 
                self.initialize_DGC_center(new_dgc) # Initialize the center position of a new DGC
                self.initialize_DGC_weights(new_dgc) # Initialize the weight of a new DGC
                self.initialize_DGC_sigmas(new_dgc) # Initialize the sigma values of a new DGC
                self.initialize_rotations(new_dgc) # Initialize the rotation matrices of a new DGC
                self.initialize_severity(new_dgc) # Initialize the severity values for Gradual local changes for a new DGC
                self.dgc.append(new_dgc) # Add the new DGC to the list               
        self.DGC_number = len(self.dgc) # Update the number of DGCs

    def adjust_variable_count(self):
        # Change in the number of variables
        updated_variable_number = self.num_of_variables + (self.rng.integers(0, 2) * 2 - 1) * self.variable_number_change_severity
        updated_variable_number = np.clip(updated_variable_number, self.min_num_of_variables, self.max_num_of_variables) # Boundary check for DGC sigma

        if updated_variable_number < self.num_of_variables: # Remove variables
            variables_to_be_removed = self.rng.choice(self.num_of_variables, abs(self.num_of_variables - updated_variable_number), replace=False)
            for dgc in self.dgc:
                dgc.center = np.delete(dgc.center, variables_to_be_removed)
                dgc.sigma = np.delete(dgc.sigma, variables_to_be_removed)
                dgc.theta_matrix = np.delete(dgc.theta_matrix, variables_to_be_removed, axis=0)
                dgc.theta_matrix = np.delete(dgc.theta_matrix, variables_to_be_removed, axis=1)
                dgc.previous_shift_direction = np.delete(dgc.previous_shift_direction, variables_to_be_removed)
                dgc.sigma_direction = np.delete(dgc.sigma_direction, variables_to_be_removed)
                dgc.rotation_direction = np.delete(dgc.rotation_direction, variables_to_be_removed, axis=0)
                dgc.rotation_direction = np.delete(dgc.rotation_direction, variables_to_be_removed, axis=1)
                dgc.rotation_matrix = self.generate_rotation_matrix(dgc.theta_matrix)

        elif updated_variable_number > self.num_of_variables:
            variables_to_be_added = np.sort(self.rng.choice(updated_variable_number, abs(self.num_of_variables - updated_variable_number), replace=False))
            for dgc in self.dgc:
                for variable_to_add in variables_to_be_added:
                    # Expand Center and Sigma by shifting elements and inserting new variable
                    new_center_value = self.min_coordinate + (self.max_coordinate - self.min_coordinate) * self.rng.random()
                    dgc.center = np.insert(dgc.center, variable_to_add, new_center_value)
                    tmp = np.insert(dgc.previous_shift_direction, variable_to_add, self.rng.standard_normal() * np.mean(dgc.previous_shift_direction))
                    dgc.previous_shift_direction = tmp / np.sqrt(np.sum(tmp**2))
                    if self.Conditioning == 0: # Keeping the condition number of DGCs as 1
                        dgc.sigma = np.append(dgc.sigma, dgc.sigma[0])
                        dgc.sigma_direction = np.append(dgc.sigma_direction, dgc.sigma_direction[0])
                    elif self.Conditioning == 1: # Conditioning is not 1 for DGCs
                        new_sigma_value = self.min_sigma + ((self.max_sigma - self.min_sigma) * self.rng.random())
                        dgc.sigma = np.insert(dgc.sigma, variable_to_add, new_sigma_value)
                        dgc.sigma_direction = np.insert(dgc.sigma_direction, variable_to_add, self.rng.integers(0, 2) * 2 - 1)
                    if self.rotation == 0: # Without rotation
                        dgc.rotation_matrix = np.eye(updated_variable_number)
                        dgc.theta_matrix = np.zeros((updated_variable_number, updated_variable_number))
                    elif self.rotation == 1: # With rotation
                        row, col = dgc.theta_matrix.shape
                        new_row = self.min_angle + (self.max_angle - self.min_angle) * self.rng.random((1, col))
                        new_col = self.min_angle + (self.max_angle - self.min_angle) * self.rng.random((row+1, 1))
                        dgc.theta_matrix = np.insert(dgc.theta_matrix, variable_to_add, new_row, axis=0)
                        dgc.theta_matrix = np.insert(dgc.theta_matrix, variable_to_add, new_col, axis=1)
                        dgc.theta_matrix = np.triu(dgc.theta_matrix, 1)
                        new_rotation_direction_row = self.rng.integers(0, 2, (1, col)) * 2 - 1
                        new_rotation_direction_col = self.rng.integers(0, 2, (row+1, 1)) * 2 - 1
                        dgc.rotation_direction = np.insert(dgc.rotation_direction, variable_to_add, new_rotation_direction_row, axis=0)
                        dgc.rotation_direction = np.insert(dgc.rotation_direction, variable_to_add, new_rotation_direction_col, axis=1)
                        dgc.rotation_direction = np.triu(dgc.rotation_direction, 1)
                dgc.rotation_matrix = self.generate_rotation_matrix(dgc.theta_matrix)

        self.num_of_variables = updated_variable_number


    def adjust_cluster_count(self):
        # Change in the number of clusters. Used when the number of clusters is defined by external parts of the system
        self.cluster_number = self.cluster_number + (self.rng.integers(0, 2) * 2 - 1) * self.cluster_number_change_severity
        self.cluster_number = np.clip(self.cluster_number, self.min_cluster_number, self.max_cluster_number) # Bound the cluster number

    

ddg = DDG()