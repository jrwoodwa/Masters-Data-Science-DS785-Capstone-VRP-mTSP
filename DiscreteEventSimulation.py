import argparse  # Argument parser for command-line options
import simpy  # Discrete Event Simulation
import numpy as np  # Vectorized operations
import pandas as pd  # Data management
import time  # Track runtime
import os  # Directory management
import pickle  # File management

class JobShop:
    def __init__(self, env, num_machines, cost_matrix, setup_df, shared_job_queue, remaining_time, job_constraints, problem_df, sort_lpt=False, sequencing_rule=None, w1=0.5, debug=False):
        self.env = env
        self.machines = [simpy.Resource(env, capacity=1) for _ in range(num_machines)]
        self.cost_matrix = cost_matrix
        self.setup_df = setup_df
        self.num_machines = num_machines
        self.num_jobs = len(job_constraints)
        self.shared_job_queue = shared_job_queue
        self.remaining_time = remaining_time.copy()
        self.job_constraints = job_constraints
        self.problem_df = problem_df
        self.sort_lpt = sort_lpt
        self.sequencing_rule = sequencing_rule
        self.debug = debug
        self.w1 = w1

        # Track job times
        self.job_tracking_df = pd.DataFrame({'Job': range(1, self.num_jobs + 1),
                                             'Start_Time': [None] * self.num_jobs,
                                             'End_Time': [None] * self.num_jobs,
                                             'Machine': [-1] * self.num_jobs})
        # Track last job processed by each machine
        self.last_job_per_machine = [0] * self.num_machines
        # Start job assignment per machine
        for machine_idx in range(self.num_machines):
            self.env.process(self.assign_next_job(machine_idx, current_job_idx=0))
        # Add jobs to queue process
        self.env.process(self.add_jobs_to_queue())

    def add_jobs_to_queue(self):
        '''Add jobs to queue when constraints are met.'''
        job_constraints = np.array(self.job_constraints)
        unique_a_values = np.sort(np.unique(job_constraints[job_constraints > 0]))
        added_jobs = set(self.shared_job_queue)
        for a_value in unique_a_values:
            if a_value > self.env.now:
                yield self.env.timeout(a_value - self.env.now)
            matching_jobs = np.where((job_constraints == a_value) &
                                     (~np.isin(np.arange(1, len(job_constraints) + 1), list(added_jobs))))[0]
            for idx in matching_jobs:
                job_id = idx + 1
                if job_id not in added_jobs:
                    self.shared_job_queue.append(job_id)
                    added_jobs.add(job_id)
                    for machine_idx in range(self.num_machines):
                        if len(self.machines[machine_idx].users) == 0:
                            last_job_idx = self.last_job_per_machine[machine_idx]
                            self.env.process(self.assign_next_job(machine_idx, current_job_idx=last_job_idx))

    def process_and_setup_job(self, current_job_idx, next_job_idx, machine_idx):
        '''Process and set up jobs.'''
        start_time = self.env.now
        total_time = self.cost_matrix[current_job_idx, next_job_idx, machine_idx]
        # Record start time
        self.job_tracking_df.loc[self.job_tracking_df['Job'] == next_job_idx, 'Start_Time'] = start_time
        self.job_tracking_df.loc[self.job_tracking_df['Job'] == next_job_idx, 'Machine'] = machine_idx
        yield self.env.timeout(total_time) # delay the machine for that time
        if self.debug: 
            print(f"[DEBUG] Total Minutes:{total_time}")
        # Record end time
        end_time = self.env.now
        self.job_tracking_df.loc[self.job_tracking_df['Job'] == next_job_idx, 'End_Time'] = end_time
        # Update last job processed
        self.last_job_per_machine[machine_idx] = next_job_idx
        # Assign next job
        self.env.process(self.assign_next_job(machine_idx, current_job_idx=next_job_idx))
    
    def get_setup_transition_times(self, from_recipe, to_recipes, machine):
        """Extract setup transition times, handling duplicates."""
        if self.debug:  # Print compact details
            print(f"[DEBUG] Machine: {machine}; From recipe: {from_recipe}; To recipes: {to_recipes}")
        
        idx = pd.IndexSlice
        transition_times = [
            self.setup_df.loc[(from_recipe, to_recipe), idx['Transition_Time_Minutes', machine]]
            for to_recipe in to_recipes
        ]
        # Ensure 1D numpy array
        transition_times = np.array(transition_times).ravel()
        
        if self.debug:  # Print compact details
            print(f"[DEBUG] Setup Minutes:{transition_times}")
        
        return transition_times
   
    def apply_sequencing_rule(self, from_recipe, available_jobs, machine_idx):
        """
        Filter jobs by sequencing rule. Includes jobs in the from and target clusters if a match is found.
        Returns unfiltered jobs if sequencing_rule is None or no target match.
        """
        # Map clusters to indices
        cluster_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    
        # Check if sequencing rule is None; if so, return all jobs
        if self.sequencing_rule is None:
            if self.debug:
                print(f"[DEBUG] No sequencing rule provided. Returning all available jobs for Machine {machine_idx+1}.")
            return available_jobs
    
        # Determine from cluster index
        from_cluster_idx = cluster_to_index[from_recipe[0]]
    
        # Get target cluster indices from sequencing rule
        target_cluster_idxs = np.where(self.sequencing_rule[from_cluster_idx, :, machine_idx] == 1)[0]
    
        # Map target indices back to cluster names
        target_clusters = [key for key, value in cluster_to_index.items() if value in target_cluster_idxs]
    
        # Include the from cluster to allow both clusters in the filtered jobs
        target_clusters.append(from_recipe[0])
    
        # Filter available jobs to include only from and target clusters
        job_clusters = self.problem_df.loc[self.problem_df['Job'].isin(available_jobs), ['Job', 'Recipe_Cluster']]
        filtered_jobs = job_clusters[job_clusters['Recipe_Cluster'].isin(target_clusters)]['Job'].tolist()
    
        # Return filtered jobs or all jobs if no matches found
        if not filtered_jobs:
            if self.debug:
                print(f"No matching jobs found for target clusters {target_clusters}. Available jobs: {available_jobs}")
            return available_jobs
        else:
            if self.debug:
                print(f"[DEBUG] Available jobs: {available_jobs}")
                print(f"[DEBUG] Filtered jobs after sequencing rule: {filtered_jobs}")
            return filtered_jobs

    
    def next_job_selection(self, current_job_idx, available_jobs, machine_idx):
        """
        Select the next job based on AI sequence rule heuristiSelectc and SPT or LPT tie-breaking.
        Applies sequencing rule if provided.
        """
        # Get from_recipe tuple
        from_recipe_row = self.problem_df[self.problem_df['Job'] == current_job_idx]
        from_recipe = tuple(from_recipe_row[['Recipe_Cluster', 'Recipe']].values[0])
        #print(f"[DEBUG] Current job index: {current_job_idx}, From recipe: {from_recipe}")
    
        # Filter jobs using sequencing rule (if self.sequencingrule==None, this skips)
        filtered_jobs = self.apply_sequencing_rule(from_recipe, available_jobs, machine_idx)
        
        # Get to_recipes tuples for filtered jobs
        to_recipes_rows = self.problem_df[self.problem_df['Job'].isin(filtered_jobs)]
        to_recipes = to_recipes_rows[['Recipe_Cluster', 'Recipe']].apply(tuple, axis=1).tolist()
        #print(f"[DEBUG] To recipes for filtered jobs: {to_recipes}")
    
        # Retrieve setup times
        setup_times = self.get_setup_transition_times(from_recipe, 
                                                      to_recipes,
                                                      machine_idx + 1) # 1-based machines for pandas dataframes
        #print(f"[DEBUG] Setup times for transitions from {from_recipe} to {to_recipes}: {setup_times}")
    
        # Apply mask for valid jobs
        valid_mask = setup_times != BIG_M
        valid_indices = np.array(filtered_jobs)[valid_mask]
        valid_costs = setup_times[valid_mask]
        # print(f"[DEBUG] Valid job indices after applying BIG_M mask: {valid_indices}")
        # print(f"[DEBUG] Valid costs after applying BIG_M mask: {valid_costs}")
    
        if valid_costs.size == 0: # if no valid jobs
            # print(f"[DEBUG] No valid jobs found. Returning None.")
            return None
    
        # Find jobs with minimum cost
        min_cost_mask = valid_costs == valid_costs.min()
        min_cost_jobs = valid_indices[min_cost_mask]
        # print(f"[DEBUG] Jobs with minimum cost: {min_cost_jobs}")
    
        if min_cost_jobs.size > 1: # Tie-breaking for multiple minimum-cost jobs
            processing_column = f"Machine{machine_idx + 1}_ProcessMinutes"
            processing_times = self.problem_df.loc[min_cost_jobs - 1, processing_column].to_numpy()
            next_job = min_cost_jobs[np.argmax(processing_times) if self.sort_lpt else np.argmin(processing_times)]
            if self.debug:
                print(f"[DEBUG] Tie-breaking ({'LPT' if self.sort_lpt else 'SPT'}); Processing minutes: {processing_times} \t Next job: {next_job}")

        else: # Single minimum-cost job
            next_job = min_cost_jobs[0]
            # print(f"[DEBUG] Single minimum-cost job. Selected next job: {next_job}")
    
        return next_job

    def assign_next_job(self, machine_idx, current_job_idx):
        '''Assign the next job to a machine.'''
        if current_job_idx == 0:
            next_job_idx = machine_idx + 1
        else:
            available_jobs = [
                job for job in self.shared_job_queue
                if self.cost_matrix[current_job_idx, job, machine_idx] != BIG_M
            ]
            if not available_jobs:
                return
            next_job_idx = self.next_job_selection(current_job_idx, available_jobs, machine_idx)
        if next_job_idx is not None:
            if next_job_idx in self.shared_job_queue:
                self.shared_job_queue.remove(next_job_idx)
            with self.machines[machine_idx].request() as request:
                yield request
                yield self.env.process(self.process_and_setup_job(current_job_idx, next_job_idx, machine_idx))

    def get_job_tracking_dataframe(self):
        '''Return job tracking DataFrame.'''
        return self.job_tracking_df

    def create_transition_matrix(self, job_tracking_df):
        """Create a transition matrix from job tracking data."""
        sorted_df = job_tracking_df.sort_values(by=['Machine', 'Start_Time']).reset_index(drop=True)
        transition_matrix = np.zeros((self.num_jobs + 1, self.num_jobs + 1, self.num_machines), dtype=int)
        
        for machine_idx in range(self.num_machines):
            machine_jobs = sorted_df[sorted_df['Machine'] == machine_idx]['Job'].values
            from_jobs = np.concatenate(([0], machine_jobs))
            to_jobs = machine_jobs
            for from_job, to_job in zip(from_jobs, to_jobs):
                if from_job == 0 and to_job == 0:
                    continue
                transition_matrix[from_job, to_job, machine_idx] += 1
        return transition_matrix
    def add_return_to_depot(self):
        '''Add return-to-depot steps in job tracking DataFrame.'''
        last_jobs = self.job_tracking_df.loc[self.job_tracking_df.groupby('Machine')['End_Time'].idxmax().dropna()]
        depot_entries = []
        for _, row in last_jobs.iterrows():
            end_time = row['End_Time']
            machine_idx = row['Machine']
            if pd.notna(end_time) and machine_idx != -1:
                total_time_to_depot = self.cost_matrix[row['Job'], 0, machine_idx]
                depot_entries.append({'Job': 0,
                                      'Start_Time': end_time,
                                      'End_Time': end_time + total_time_to_depot,
                                      'Machine': machine_idx})
        depot_df = pd.DataFrame(depot_entries)
        self.job_tracking_df = pd.concat([self.job_tracking_df, depot_df], ignore_index=True).sort_values(by=['Machine', 'Start_Time']).reset_index(drop=True)
    
    def simulate(self, sim_time):
        """Main Method
        Run the job shop simulation."""
        print('Job Shop Simulation Start')
        start_time = time.time()
        self.env.run(until=sim_time)
        end_time = time.time()
        
        self.job_tracking_df = self.get_job_tracking_dataframe().sort_values(by=['Machine', 'Start_Time']).reset_index(drop=True)
        self.add_return_to_depot()
        elapsed_time_seconds = end_time - start_time
        print(f'Simulation computation time: {elapsed_time_seconds:.2f} seconds')
        print('Job Shop Simulation End')
        
        return self.get_solution(elapsed_time_seconds)
    
    def get_solution(self, runtime_seconds):
        """Compute max flow time, average makespan, and prepare data for saving."""
        transition_matrix = self.create_transition_matrix(self.job_tracking_df)
        max_flow_time = np.max(self.job_tracking_df['End_Time'])
        avg_makespan = np.sum(self.cost_matrix * transition_matrix)/self.num_machines
        w1 = self.w1
        obj_value = w1 * max_flow_time + (1 - w1) * avg_makespan
        
        # Gather solution data for saving
        solution_data = {
            'max_flow_time': max_flow_time,
            'avg_makespan': avg_makespan,
            'obj_value': obj_value,
            'runtime_seconds': round(runtime_seconds,2),
            'job_tracking_df': self.job_tracking_df,
            'transition_matrix': transition_matrix
        }
        
        return solution_data


# Save solution data in one file
def save_results(solution_data, queue_size, problem_number, solution_number, algorithm_type, training):
    '''
    Save the entire solution_data dictionary in one file.
    '''
    if training: 
        str = 'Training_'
    else:
        str = ''
    output_folder = f'ProblemSets/{str}Queue{queue_size}/Solutions/{algorithm_type}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    base_filename = f"Problem_{problem_number}_Solution_{solution_number}.pkl"
    output_path = f"{output_folder}/{base_filename}"
    
    # Save entire solution_data dictionary
    with open(output_path, 'wb') as file:
        pickle.dump(solution_data, file)
    print(f"Solution data saved at: {output_path}")

# Main script entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Job Shop Simulation")
    parser.add_argument('--QueueSize', type=int, required=True, help="Queue size of jobs.")
    parser.add_argument('--TrainingProblemSet', type=str, required=True, help="Set 'True' for LPT or 'False' for SPT.")
    parser.add_argument('--ProblemNumber', type=int, required=True, help="Problem number to use.")
    parser.add_argument('--SolutionNumber', type=int, required=True, help="Solution number for saving results.")
    parser.add_argument('--AlgorithmType', type=str, required=True, help="Algorithm type for saving results.")
    parser.add_argument('--sort_lpt', type=str, required=True, help="Set 'True' for LPT or 'False' for SPT.")
    parser.add_argument('--ArrivalTimes_expected', type=str, required=True, help="Set 'True' for Expected Arrivals, a, or 'False' for Known Arrival times, a_true (AKA GodMode).")
    parser.add_argument('--sequencing_rule', type=str, default="None", help="String representation of a numpy array for sequencing rule, or 'None'.")

    args = parser.parse_args()

    ###
    if args.ArrivalTimes_expected.lower() == 'true':
        arrival_times_column = 'a'
    else:
        arrival_times_column = 'a_true'

    #print(f"[DEBUG] {arrival_times_column}")

    # Convert to boolean
    sort_lpt = args.sort_lpt.lower() == 'true'
    isTraining = args.TrainingProblemSet.lower() == 'true'

    # Interpret sequencing_rule as a numpy array or None
    if args.sequencing_rule.lower() != 'none':
        sequencing_rule = np.array(eval(args.sequencing_rule))
        print(f"Sequencing rule set to numpy array: {sequencing_rule}")
    else:
        sequencing_rule = None
        print("No sequencing rule provided; defaulting to None.")

    if isTraining: 
        str = 'Training_'
    else:
        str = ''
    
    # Load problem and setup data
    problem_df = pd.read_pickle(f'ProblemSets/{str}Queue{args.QueueSize}/Problem_{args.ProblemNumber}_jobs.pkl')
    cost_matrix = np.load(f'ProblemSets/{str}Queue{args.QueueSize}/Problem_{args.ProblemNumber}_cost_matrix.npy')
    setup_df = pd.read_pickle(f"RecipeToRecipe_Setups.pkl").unstack(level='Machine')

    # Constants
    NUM_MACHINES = cost_matrix.shape[2]
    SIM_TIME = 10000
    BIG_M = 10000
    INITIAL_REMAINING_TIME = np.array(problem_df['Remaining_Minutes'].dropna().to_list(), dtype=int)

    # Initialize environment and JobShop
    env = simpy.Environment()
    shared_job_queue = [job for job in problem_df['Job'].tolist() if problem_df[arrival_times_column].iloc[job - 1] == 0]
    job_constraints = problem_df[arrival_times_column].tolist()

    # Instantiate JobShop
    job_shop = JobShop(
        env, NUM_MACHINES, cost_matrix, setup_df, shared_job_queue,
        INITIAL_REMAINING_TIME, job_constraints, problem_df,
        sort_lpt=sort_lpt, 
        sequencing_rule=sequencing_rule, 
        debug=False
    )

    # Run simulation and get solution
    solution_data = job_shop.simulate(SIM_TIME)

    # Save the results using solution_data
    save_results(solution_data,  # data dict output
                 args.QueueSize, args.ProblemNumber, args.SolutionNumber, args.AlgorithmType, isTraining) # the rest determines folder directory
