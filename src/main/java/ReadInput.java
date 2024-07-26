package main.java;

import java.io.*;
import java.util.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.stream.Collectors;


public class ReadInput {

    public Map<String, Object> data;

    public int[][] solution;

    public ReadInput() {
        data = new HashMap<String, Object>();
    }

    // Function to initialise the solution array
    public int[][] initialiseSolution() {
        int numCaches = (Integer) data.get("number_of_caches");
        int numVideos = (Integer) data.get("number_of_videos");

        return solution = new int[numCaches][numVideos];
    }

    // Function to check if a given solution is valid based on capacity constraints
    public boolean isValidSolution(int[][] solution) {
        int numCaches = (Integer) data.get("number_of_caches");
        int cacheSize = (Integer) data.get("cache_size");
        int[] videoSizes = (int[]) data.get("video_size_desc");

        for (int i = 0; i < numCaches; i++) {
            int cacheSizeUsed = 0;
            for (int j = 0; j < videoSizes.length; j++) {
                cacheSizeUsed += solution[i][j] * videoSizes[j];
            }
            if (cacheSizeUsed > cacheSize) {
                return false; // Break if cache size is exceeded
            }
        }
        return true;

    }

    public int fitness(int[][] solution) {
        // Store the data from the input
        int[] videoSizes = (int[]) data.get("video_size_desc");
        int cacheSize = (Integer) data.get("cache_size");
        List<List<Integer>> epToCacheLatency = (List<List<Integer>>) data.get("ep_to_cache_latency");
        Map<String, String> videoEdRequest = (Map<String, String>) data.get("video_ed_request");

        // Check for cache overflow (return -1 if capacity exceeded)
        for (int cacheIndex = 0; cacheIndex < solution.length; cacheIndex++) {
            int cacheSizeUsed = 0;
            for (int videoIndex = 0; videoIndex < solution[cacheIndex].length; videoIndex++) {
                if (solution[cacheIndex][videoIndex] == 1) {
                    cacheSizeUsed += videoSizes[videoIndex];
                }
            }
            if (cacheSizeUsed > cacheSize) {
                return -1;
            }
        }

        // Calculate the total gain and total requests
        double totalGain = 0.0;
        int totalRequests = 0;

        for (String key : videoEdRequest.keySet()) {
            String[] parts = key.split(", ");
            int videoIndex = Integer.parseInt(parts[0].split(" ")[1]);
            int endpointId = Integer.parseInt(parts[1].split(" ")[1]);
            int numRequests = Integer.parseInt(videoEdRequest.get(key).split(" ")[0]);
            
            int dcLatency = ((List<Integer>) data.get("ep_to_dc_latency")).get(endpointId);
            int dcCost = dcLatency;
            
            // Find minimum cost of downloading from a connected cache
            int minCacheCost = dcCost;
            for (int cacheIndex = 0; cacheIndex < solution.length; cacheIndex++) {
                // Update cost if video is stored in cache and latency is lower
                if (solution[cacheIndex][videoIndex] == 1 && epToCacheLatency.get(endpointId).get(cacheIndex) < minCacheCost) {
                    minCacheCost = epToCacheLatency.get(endpointId).get(cacheIndex);
                }
            }

            // Calculate gain for this request
            if (minCacheCost < dcCost) {
                double gain = (dcCost - minCacheCost);
                totalGain += gain * numRequests;
            }
            totalRequests += numRequests;
        }

        // Calculate total fitness score
        double fitnessScore = (totalGain / totalRequests) * 1000;
        return (int) fitnessScore;
    }

    // Function that performs hill-climbing algorithm on the solution
    public void hillClimbing() {
        // Initialise solution
        initialiseSolution();
        // double bestScore = fitness(solution);
        double bestScore = 0.0;

        // Loop for iterative improvement
        boolean improvement = true;
        int iteration = 0;

        while (improvement) {
            improvement = false;
            double iterationBestScore = bestScore;
            int[] bestMove = new int[] { -1, -1 }; // Stores the best move

            for (int cacheIndex = 0; cacheIndex < solution.length; cacheIndex++) {
                for (int videoIndex = 0; videoIndex < solution[cacheIndex].length; videoIndex++) {
                    
                    if (solution[cacheIndex][videoIndex] == 0) {
                        solution[cacheIndex][videoIndex] = 1; // Make a move
                        double neighbourScore = fitness(solution);
                        if (neighbourScore > iterationBestScore) {
                            iterationBestScore = neighbourScore;
                            bestMove[0] = cacheIndex;
                            bestMove[1] = videoIndex;
                            improvement = true;
                        }
                        solution[cacheIndex][videoIndex] = 0; // Revert move for next iteration
                    }
                    
                }
            }

            // If improvement was found, make the best move permanent
            if (bestMove[0] != -1) {
                solution[bestMove[0]][bestMove[1]] = 1;
                bestScore = iterationBestScore; // Update global best score
                System.out.println("Iteration " + iteration + ": " + bestScore);
            }
            iteration++;
        }

        System.out.println("Final best score after hill climbing: " + bestScore);
    }

    // Helper method to print the current state of the solution
    public void printSolutionState() {
        System.out.println("Current solution state:");
        for (int[] row : solution) {
            for (int cell : row) {
                System.out.print(cell + " ");
            }
            System.out.println(); // New line after each row for better readability
        }
    }

    // Genetic Algorithm Methods

    private List<int[][]> generateRandomPopulation(int populationSize, int numCaches, int numVideos) {
        List<int[][]> population = new ArrayList<>();
        Random random = new Random();
    
        int[] videoSizes = (int[]) data.get("video_size_desc");
        int cacheSize = (Integer) data.get("cache_size");
    
        for (int i = 0; i < populationSize; ) {
            int[][] individual = new int[numCaches][numVideos];
    
            // Strategy: Fill each cache up to a maximum capacity threshold to avoid invalid solutions
            for (int cacheIndex = 0; cacheIndex < numCaches; cacheIndex++) {
                int capacityUsed = 0;
                for (int videoIndex = 0; videoIndex < numVideos; videoIndex++) {
                    // Randomly decide to add the video based on a probability and if it fits
                    if (random.nextDouble() < 0.3 && capacityUsed + videoSizes[videoIndex] <= cacheSize) {
                        individual[cacheIndex][videoIndex] = 1;
                        capacityUsed += videoSizes[videoIndex];
                    } else {
                        individual[cacheIndex][videoIndex] = 0;
                    }
                }
            }
    
            // Use isValidSolution to double-check the generated individual; it should always return true by design
            if (isValidSolution(individual)) {
                population.add(individual);
                i++; // Ensure this is incremented only when a valid individual is added
            }
            // Removed the else block that decrements i to avoid infinite loop
        }
    
        return population;
    }

    public List<int[][]> generateInitialPopulation(int numCaches, int numVideos) {
        Random random = new Random();
        List<int[][]> population = new ArrayList<>();
        int populationSize = 50; // Assuming a population size of 50 as specified
        int[] videoSizes = (int[]) data.get("video_size_desc");
        int cacheSize = (Integer) data.get("cache_size");

        while (population.size() < populationSize) {
            int[][] individual = new int[numCaches][numVideos];

            for (int i = 0; i < numCaches; i++) {
                int currentCacheSize = 0;
                for (int j = 0; j < numVideos; j++) {
                    // Use a heuristic or random choice to add a video to the cache
                    if (random.nextBoolean() && currentCacheSize + videoSizes[j] <= cacheSize) {
                        individual[i][j] = 1;
                        currentCacheSize += videoSizes[j];
                    } else {
                        individual[i][j] = 0;
                    }
                }
            }

            // Check if the generated individual is valid
            if (isValidSolution(individual)) {
                population.add(individual);
            } else {
                // If not valid, consider a fallback strategy to adjust the individual or simply try again
                // Here, we simply try again by continuing the loop without action
            }
        }

        return population;
    }

    // Generates an initial population using a Greedy approach
    public List<int[][]> generateGreedyInitialPopulation(int numCaches, int numVideos, int populationSize) {
        List<int[][]> population = new ArrayList<>();
        int[] videoSizes = (int[]) data.get("video_size_desc"); // Assuming video sizes are provided in 'data'
        int cacheSize = (Integer) data.get("cache_size"); // Assuming cache size is provided in 'data'

        for (int p = 0; p < populationSize; p++) {
            int[][] individual = new int[numCaches][numVideos];
            
            for (int cacheIndex = 0; cacheIndex < numCaches; cacheIndex++) {
                List<Integer> videosConsidered = new ArrayList<>();
                int currentCacheSize = 0;

                // Sort or prioritize videos based on a heuristic (e.g., smallest first, most popular, etc.)
                // For simplicity, this example randomly orders videos but consider using a specific heuristic
                List<Integer> videoOrder = generateRandomVideoOrder(numVideos);
                
                for (Integer videoIndex : videoOrder) {
                    if (currentCacheSize + videoSizes[videoIndex] <= cacheSize) {
                        individual[cacheIndex][videoIndex] = 1; // Add video to cache
                        currentCacheSize += videoSizes[videoIndex];
                    }
                }
            }
            if (isValidSolution(individual)) {
                population.add(individual);
            }
            // Optionally, handle the case where an individual is not valid
        }

        return population;
    }

    private List<Integer> generateRandomVideoOrder(int numVideos) {
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < numVideos; i++) {
            order.add(i);
        }
        java.util.Collections.shuffle(order);
        return order;
    }
    

    public List<int[][]> crossover(int[][] parent1, int[][] parent2, double crossoverProbability, int currentGeneration, int numGenerations) {
        List<int[][]> children = new ArrayList<>();
        Random random = new Random();

        // Check if crossover happens, based on crossover probability
        if (random.nextDouble() < crossoverProbability) {
            // Find crossover point
            int numCaches = parent1.length;
            int numVideos = parent1[0].length;

            // Selects a fixed crossover point in the middle of the array
            // int crossoverPoint = numVideos / 2;

            // Dynamically selects a crossover point within the array bounds
            int crossoverPoint = 1 + random.nextInt(numVideos - 2);

            // Adaptive crossover point selection
            // int crossoverPoint;
            // if (currentGeneration < numGenerations / 2) {
            //     // Early generations: more exploratory
            //     crossoverPoint = 1 + random.nextInt(numVideos - 2);
            // } else {
            //     // Mid generations: balanced approach, selects a point based on a Gaussian distribution
            //     crossoverPoint = (int) (random.nextGaussian() * (numVideos / 4) + (numVideos / 2));
            //     crossoverPoint = Math.min(Math.max(crossoverPoint, 1), numVideos - 2); // Ensure it's within bounds
            // }

            int[][] child1 = new int[numCaches][numVideos];
            int[][] child2 = new int[numCaches][numVideos];

            // Create children by swapping parts of each parent
            for (int i = 0; i < numCaches; i++) {
                for (int j = 0; j < numVideos; j++) {
                    if (j < crossoverPoint) {
                        child1[i][j] = parent1[i][j];
                        child2[i][j] = parent2[i][j];
                    } else {
                        child1[i][j] = parent2[i][j];
                        child2[i][j] = parent1[i][j];
                    }
                }
            }

            // Add children to list if they are valid, otherwise discard
            if (isValidSolution(child1)) {
                children.add(child1);
            }
            if (isValidSolution(child2)) {
                children.add(child2);
            }

        }

        return children;
    }

    public void mutate(int[][] individual, double mutationProbability) {
        Random random = new Random();
        int numCaches = individual.length;
        int numVideos = individual[0].length;

        for (int i = 0; i < numCaches; i++) {
            for (int j = 0; j < numVideos; j++) {
                if (random.nextDouble() < mutationProbability) {
                    individual[i][j] = individual[i][j] == 0 ? 1 : 0;

                    // Check if the mutation is valid
                    if (!isValidSolution(individual)) {
                        individual[i][j] = 1 - individual[i][j];
                   }
                }
            }
        }
    }

    public void applyMutation(List<int[][]> population, double mutationProbability) {
        if (population.isEmpty()) {
            return;
        }
        // double mutationProbability = 1.0 / (population.size());
        // Optimal mutation rate 
        //double mutationProbability = 0.01;

        for (int[][] individual : population) {
            mutate(individual, mutationProbability);
        }
    }

    public List<int[][]> cullToFittest(List<int[][]> combinedPopulation, int targetPopulationSize) {
        // Sort combined population based on fitness scores
        combinedPopulation.sort((individual1, individual2) -> Double.compare(fitness(individual2), fitness(individual1)));

        // Check if culling is needed
        if (combinedPopulation.size() <= targetPopulationSize) {
            return new ArrayList<int[][]>(combinedPopulation);
        } 

        // Select the fittest individuals
        List<int[][]> fittestPopulation = new ArrayList<int[][]>(combinedPopulation.subList(0, targetPopulationSize));
        return fittestPopulation;
    }

    private double bestFitness;

    public void geneticAlgorithm(int populationSize, int numCaches, int numVideos, int numGenerations, int targetPopulationSize, double crossoverProbability, double mutationProbability) {
        // Generate the initial population
         List<int[][]> population = generateRandomPopulation(populationSize, numCaches, numVideos);
        // List<int[][]> population = generateInitialPopulation(numCaches, numVideos);
        // List<int[][]> population = generateGreedyInitialPopulation(numCaches, numVideos, populationSize);
        // List<int[][]> population = generate_initial_population();

        // List<int[][]> population = generateRandomOldPopulation(populationSize, numCaches, numVideos);

        // Perform genetic algorithm for a number of generations
        for (int generation = 0; generation < numGenerations; generation++) {
            List<int[][]> newGeneration = new ArrayList<>();

            // Perform crossover
            for (int i = 0; i < population.size() - 1; i += 2) {
                List<int[][]> children = crossover(population.get(i), population.get(i + 1), crossoverProbability, generation, numGenerations);
                newGeneration.addAll(children);
            }

            // Apply mutation to children
            applyMutation(newGeneration, mutationProbability);

            // Combine parents and children
            List<int[][]> combinedPopulation = new ArrayList<int[][]>(population);
            combinedPopulation.addAll(newGeneration);

            // Cull to fittest individuals
            population = cullToFittest(combinedPopulation, targetPopulationSize);
            
            System.out.println("Generation " + generation + " fitness: " + fitness(population.get(0)));
        }
        bestFitness = fitness(population.get(0));
        System.out.println("Best fitness: " + bestFitness);
    }

    public void runParameterEvaluationExperiments(int numExperiments, int populationSize, int numCaches, int numVideos, int numGenerations, int targetPopulationSize) {
        double[] crossoverProbabilities = new double[]{0.1, 0.3, 0.5, 0.7, 0.9};
        double[] mutationRates = new double[]{0.001, 0.01, 0.05, 0.1, 0.15};
        
        List<String> results = new ArrayList<>();
        results.add("Crossover Probability,Mutation Rate,Best Fitness");
    
        for (double crossoverProbability : crossoverProbabilities) {
            for (double mutationRate : mutationRates) {
                double avgBestFitness = 0;
                for (int experiment = 0; experiment < numExperiments; experiment++) {
                    geneticAlgorithm(populationSize, numCaches, numVideos, numGenerations, targetPopulationSize, crossoverProbability, mutationRate);
                    // Directly use the lastBestFitness updated by the geneticAlgorithm method
                    avgBestFitness += bestFitness;
                }
                avgBestFitness /= numExperiments;
                results.add(crossoverProbability + "," + mutationRate + "," + avgBestFitness);
            }
        }
        
        // Write results to CSV
        try {
            Files.write(Paths.get("genetic_algorithm_parameter_evaluation.csv"), results);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Fields for PSO
    public List<int[][]> swarm;
    public List<double[][]> velocity;
    public List<Double> particleBestFitness;
    public List<int[][]> particleBestPosition;

    public void initialiseSwarm(int swarmSize, int numCaches, int numVideos) {
        // Assuming each particle is represented by an int[][] matrix similar to your solution structure
        int[] videoSizes = (int[]) data.get("video_size_desc");
        int cacheSize = (Integer) data.get("cache_size");
        this.swarm = new ArrayList<int[][]>();
        this.velocity = new ArrayList<double[][]>();
        this.particleBestFitness = new ArrayList<Double>();
        this.particleBestPosition = new ArrayList<int[][]>();
    
        for (int i = 0; i < swarmSize; i++) {
            int[][] particle = new int[numCaches][numVideos];
            double[][] particleVelocity = new double[numCaches][numVideos];
    
            // Initialize particle with a random valid solution
            for (int cacheIndex = 0; cacheIndex < numCaches; cacheIndex++) {
                int cacheCapacityUsed = 0;
                for (int videoIndex = 0; videoIndex < numVideos; videoIndex++) {
                    if (Math.random() < 0.5 && cacheCapacityUsed + videoSizes[videoIndex] <= cacheSize) {
                        particle[cacheIndex][videoIndex] = 1;
                        cacheCapacityUsed += videoSizes[videoIndex];
                    } else {
                        particle[cacheIndex][videoIndex] = 0;
                    }
                    // Initialize velocity
                    particleVelocity[cacheIndex][videoIndex] = Math.random() * 0.1 - 0.05; // Small initial velocity
                }
            }
    
            swarm.add(particle);
            velocity.add(particleVelocity);
            particleBestFitness.add(Double.NEGATIVE_INFINITY);
            particleBestPosition.add(particle.clone());
        }
    }

    public void updateParticle(int particleIndex, double w, double c1, double c2, int[][] globalBestPosition, int numCaches, int numVideos) {
        Random rand = new Random();
        int[][] particle = swarm.get(particleIndex);
        double[][] particleVelocity = velocity.get(particleIndex);
        
        for (int cacheIndex = 0; cacheIndex < numCaches; cacheIndex++) {
            for (int videoIndex = 0; videoIndex < numVideos; videoIndex++) {
                double r1 = rand.nextDouble();
                double r2 = rand.nextDouble();
    
                // Update velocity
                particleVelocity[cacheIndex][videoIndex] = w * particleVelocity[cacheIndex][videoIndex] +
                        c1 * r1 * (particleBestPosition.get(particleIndex)[cacheIndex][videoIndex] - particle[cacheIndex][videoIndex]) +
                        c2 * r2 * (globalBestPosition[cacheIndex][videoIndex] - particle[cacheIndex][videoIndex]);
    
                // Update position based on velocity
                if (Math.random() < Math.abs(particleVelocity[cacheIndex][videoIndex])) { // Simplified binary decision
                    particle[cacheIndex][videoIndex] = 1 - particle[cacheIndex][videoIndex]; // Flip bit
                }
    
                // Ensure the solution remains valid (respect cache size constraint)
                // You might need to adjust this to ensure the total size does not exceed the cache size
                if (!isValidSolution(particle)) {
                    particle[cacheIndex][videoIndex] = 1 - particle[cacheIndex][videoIndex]; // Revert change
                }
            }
        }
    }

    public void runPSOptimisation(int swarmSize, int numIterations, double w, double c1, double c2, int numCaches, int numVideos) {
        initialiseSwarm(swarmSize, numCaches, numVideos);
        int[][] globalBestPosition = new int[numCaches][numVideos]; // Best position found by any particle
        double globalBestFitness = Double.NEGATIVE_INFINITY;
    
        for (int iter = 0; iter < numIterations; iter++) {
            for (int i = 0; i < swarmSize; i++) {
                int[][] particle = swarm.get(i);
                double fitness = fitness(particle);
    
                // Update personal and global bests
                if (fitness > particleBestFitness.get(i)) {
                    particleBestFitness.set(i, fitness);
                    particleBestPosition.set(i, deepCopy(particle));
                }
                if (fitness > globalBestFitness) {
                    globalBestFitness = fitness;
                    globalBestPosition = deepCopy(particle);
                }
    
                updateParticle(i, w, c1, c2, globalBestPosition, numCaches, numVideos);
            }
            System.out.println("Iteration " + iter + " Best Fitness: " + globalBestFitness);
        }
    }

    // Copy method for 2D Arrays
    private int[][] deepCopy(int[][] original) {
        if (original == null) {
            return null;
        }

        final int[][] result = new int[original.length][];
        for (int i = 0; i < original.length; i++) {
            result[i] = Arrays.copyOf(original[i], original[i].length);
        }
        return result;
    }

    public void readGoogle(String filename) throws IOException {
             
        BufferedReader fin = new BufferedReader(new FileReader(filename));
    
        String system_desc = fin.readLine();
        String[] system_desc_arr = system_desc.split(" ");
        int number_of_videos = Integer.parseInt(system_desc_arr[0]);
        int number_of_endpoints = Integer.parseInt(system_desc_arr[1]);
        int number_of_requests = Integer.parseInt(system_desc_arr[2]);
        int number_of_caches = Integer.parseInt(system_desc_arr[3]);
        int cache_size = Integer.parseInt(system_desc_arr[4]);
    
        Map<String, String> video_ed_request = new HashMap<String, String>();
        String video_size_desc_str = fin.readLine();
        String[] video_size_desc_arr = video_size_desc_str.split(" ");
        int[] video_size_desc = new int[video_size_desc_arr.length];
        for (int i = 0; i < video_size_desc_arr.length; i++) {
            video_size_desc[i] = Integer.parseInt(video_size_desc_arr[i]);
        }
    
        List<List<Integer>> ed_cache_list = new ArrayList<List<Integer>>();
        List<Integer> ep_to_dc_latency = new ArrayList<Integer>();
        List<List<Integer>> ep_to_cache_latency = new ArrayList<List<Integer>>();
        for (int i = 0; i < number_of_endpoints; i++) {
            ep_to_dc_latency.add(0);
            ep_to_cache_latency.add(new ArrayList<Integer>());
    
            String[] endpoint_desc_arr = fin.readLine().split(" ");
            int dc_latency = Integer.parseInt(endpoint_desc_arr[0]);
            int number_of_cache_i = Integer.parseInt(endpoint_desc_arr[1]);
            ep_to_dc_latency.set(i, dc_latency);
    
            for (int j = 0; j < number_of_caches; j++) {
                ep_to_cache_latency.get(i).add(ep_to_dc_latency.get(i) + 1);
            }
    
            List<Integer> cache_list = new ArrayList<Integer>();
            for (int j = 0; j < number_of_cache_i; j++) {
                String[] cache_desc_arr = fin.readLine().split(" ");
                int cache_id = Integer.parseInt(cache_desc_arr[0]);
                int latency = Integer.parseInt(cache_desc_arr[1]);
                cache_list.add(cache_id);
                ep_to_cache_latency.get(i).set(cache_id, latency);
            }
            ed_cache_list.add(cache_list);
        }
    
        for (int i = 0; i < number_of_requests; i++) {
            String[] request_desc_arr = fin.readLine().split(" ");
            String video_id = request_desc_arr[0];
            String ed_id = request_desc_arr[1];
            String requests = request_desc_arr[2];
            video_ed_request.put("video " + video_id + ", end " + ed_id, requests + " requests");
        }
    
        data.put("number_of_videos", number_of_videos);
        data.put("number_of_endpoints", number_of_endpoints);
        data.put("number_of_requests", number_of_requests);
        data.put("number_of_caches", number_of_caches);
        data.put("cache_size", cache_size);
        data.put("video_size_desc", video_size_desc);
        data.put("ep_to_dc_latency", ep_to_dc_latency);
        data.put("ep_to_cache_latency", ep_to_cache_latency);
        data.put("ed_cache_list", ed_cache_list);
        data.put("video_ed_request", video_ed_request);
    
        fin.close();
     
     }

     public String toString() {
        String result = "";

        //for each endpoint: 
        for(int i = 0; i < (Integer) data.get("number_of_endpoints"); i++) {
            result += "endpoint number " + i + "\n";
            //latendcy to DC
            int latency_dc = ((List<Integer>) data.get("ep_to_dc_latency")).get(i);
            result += "latency to dc " + latency_dc + "\n";
            //for each cache
            for(int j = 0; j < ((List<List<Integer>>) data.get("ep_to_cache_latency")).get(i).size(); j++) {
                int latency_c = ((List<List<Integer>>) data.get("ep_to_cache_latency")).get(i).get(j); 
                result += "latency to cache number " + j + " = " + latency_c + "\n";
            }
        }

        return result;
    }

    public static void main(String[] args) throws IOException {  
        ReadInput ri = new ReadInput();
        ri.readGoogle("input/me_at_the_zoo.in");

        // ri.initialiseSolution();
        // ri.hillClimbing();


        // Parameters for genetic algorithm
        int targetPopulationSize = 50; // Target population size
        int populationSize = 50; // Size of the population
        int numGenerations = 500; // Number of generations
        int numCaches = (Integer) ri.data.get("number_of_caches");
        int numVideos = (Integer) ri.data.get("number_of_videos");

        // ri.geneticAlgorithm(populationSize, numCaches, numVideos, numGenerations, targetPopulationSize, 0.6, 0.01);

        // Number of experiments per parameter set for averaging
        int numExperiments = 5;

        // Run parameter evaluation experiments
        // ri.runParameterEvaluationExperiments(numExperiments, populationSize, numCaches, numVideos, numGenerations, targetPopulationSize);

        // Define PSO parameters
        int swarmSize = 100; // Size of the swarm
        int numIterations = 300; // Number of iterations for PSO
        double w = 0.6; // Inertia weight
        double c1 = 2.0; // Cognitive coefficient
        double c2 = 2.0; // Social coefficient

        // Run PSO
        ri.runPSOptimisation(swarmSize, numIterations, w, c1, c2, numCaches, numVideos);

        // Manual representation of solution for Google Hashcode Example 
        // ri.solution[1][3] = 1; // Video 3 stored in cache 1
        // ri.solution[2][1] = 1; // Video 1 stored in cache 2

        // System.out.println(ri.data.get("number_of_videos") + " videos, " + ri.data.get("number_of_endpoints") + " endpoints, " + ri.data.get("number_of_requests") + " request descriptions, " + ri.data.get("number_of_caches") + " caches " + ri.data.get("cache_size") + " MB each.");
        // System.out.println(ri.data.get("video_ed_request"));
        // System.out.println(ri.toString());
        
        // Call fitness function
        // double score = ri.fitness(ri.solution);
        // System.out.println("Fitness score: " + score);
    }
}
