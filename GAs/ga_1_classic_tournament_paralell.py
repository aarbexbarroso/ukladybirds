import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import random
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# Function to load data
def load_data(file_path):
    """
    Loads data from the CSV file and separates it into ID, true labels, and predictions from both models.
    """
    data = pd.read_csv(file_path)
    ids = data.iloc[:, 0].values  # Record IDs
    y_true = data.iloc[:, 1].values  # True outputs
    model_1_probs = data.iloc[:, 2:20].values  # Model 1 probabilities
    model_2_probs = data.iloc[:, 20:38].values  # Model 2 probabilities
    return ids, y_true, model_1_probs, model_2_probs

# Evaluation function
def evaluate(y_true, combined_probs):
    """
    Evaluates accuracy by combining predictions and computing accuracy.
    """
    y_pred = np.argmax(combined_probs, axis=1)
    return accuracy_score(y_true, y_pred)

# Combination of model probabilities
def combine_probabilities(model_1_probs, model_2_probs, weight):
    """
    Combines probabilities from both models using class-specific weights.
    """
    return weight * model_1_probs + (1 - weight) * model_2_probs

# Function to create initial population
def create_population(size):
    """
    Creates an initial population of random weights for each class.
    """
    return np.random.rand(size)

# Selection of best individuals
def select(population, scores, num_parents):
    """
    Selects the best individuals from the population based on their score.
    """
    parents = []
    pop_size = len(population)
    
    for _ in range(num_parents):
        id1, id2 = np.random.randint(0, pop_size, size=2)
        if scores[id1] >= scores[id2]:
            parents.append(population[id1])
        else:
            parents.append(population[id2])
    return np.array(parents)
    
# Selection of best individuals
def reduce_pop(population, scores, num_parents):
    """
    Selects the best individuals from the population based on their score.
    """
    selected_indices = np.argsort(scores)[-num_parents:]
    return population[selected_indices]


# Crossover function
def crossover(parents, offspring_size):
    """
    Performs crossover between parents to generate new individuals.
    """
    offspring = []
    for _ in range(offspring_size):
        parent1, parent2 = random.sample(list(parents), 2)
        child = (parent1 + parent2) / 2
        offspring.append(child)
    return np.array(offspring)

# Mutation function
def mutate(offspring, mutation_rate=0.1):
    """
    Applies mutation to individuals in the population.
    """
    for i in range(len(offspring)):
        if random.random() < mutation_rate:
            offspring[i] += np.random.normal(0, 0.1)  
            offspring[i] = np.clip(offspring[i], 0, 1)  # Keeps weight between 0 and 1
    return offspring

# Main genetic algorithm
def genetic_algorithm(y_true, model_1_probs, model_2_probs, num_generations=50, population_size=100, num_parents=100, mutation_rate=0.5):
    """
    Runs the genetic algorithm to find the best class-wise weights.
    """
    
    population = create_population(population_size)  # Initial population
    scores = []
    for weight in population:
        combined_probs = combine_probabilities(model_1_probs, model_2_probs, weight)
        accuracy = evaluate(y_true, combined_probs)
        scores.append(accuracy)
    scores = np.array(scores)
    
    acc_plot = []
    acc_plot_t = []
    for generation in range(num_generations):
        
        parents = select(population, scores, num_parents)
        
        # Crossover and mutation to generate new population
        offspring_size = num_parents
        offspring = crossover(parents, offspring_size)
        offspring = mutate(offspring, mutation_rate)
        
        population = reduce_pop(population, scores, 5)
        
        scores_off = []
        for weight in offspring:
            combined_probs = combine_probabilities(model_1_probs, model_2_probs, weight)
            accuracy = evaluate(y_true, combined_probs)
            scores_off.append(accuracy)
        scores_off = np.array(scores_off)
       
        offspring = reduce_pop(offspring, scores_off, population_size - 5)
                
        # Update population
        population = np.concatenate((population, offspring))
        
        
        scores = []
        for weight in population:
            combined_probs = combine_probabilities(model_1_probs, model_2_probs, weight)
            accuracy = evaluate(y_true, combined_probs)
            scores.append(accuracy)
        
        # Selection of best individuals
        scores = np.array(scores)
        
        # Best individual of this generation
        if (generation == 0):
            best_score = max(scores)
            best_ind = np.argmax(scores)
            best_weight = population[best_ind]
        elif (max(scores) > best_score):
            best_score = max(scores)
            best_ind = np.argmax(scores)
            best_weight = population[best_ind]
        print(f"Generation {generation + 1} - Best accuracy: {best_score:.4f}")
    
    # Best set of weights found
    print(f"Generation {generation + 1} - Best accuracy: {best_score:.4f}")
    return best_weight

def run_task(SEED, model, BATCH, class_cols):
    random.seed(SEED)
    np.random.seed(SEED)
    print(f' Starting SEED {SEED} for model {model}')
    # Load data
    file_path = f'/path/ladybirds/predictions_for_ga/{model}/Batch_{BATCH}/val_predictions.csv'  
    ids, y_true, model_1_probs, model_2_probs = load_data(file_path)
    # Run genetic algorithm
    best_weights = genetic_algorithm(y_true, model_1_probs, model_2_probs)
    print(f"Best weights found for each class: {best_weights}")
    combined_probs = combine_probabilities(model_1_probs, model_2_probs, best_weights)
    accuracy_val = evaluate(y_true, combined_probs)
    ###########################################################################################################
    # Load data
    file_path = f'/path/ladybirds/predictions_for_ga/{model}/Batch_{BATCH}/test_predictions.csv'  
    ids, y_true, model_1_probs, model_2_probs = load_data(file_path)
    combined_probs = combine_probabilities(model_1_probs, model_2_probs, best_weights)
    accuracy_test = evaluate(y_true, combined_probs)
    df_final = pd.DataFrame(combined_probs, columns=class_cols)
    df_final.insert(0, "image_id", ids)
    df_final["true_label"] = y_true
    output_csv = f'/path/ladybirds/ga_outputs/ga_1_weight_200/{model}/Batch{BATCH}/predictions/SEED{SEED}_predictions.csv'
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_final.to_csv(output_csv, index=False)
    print(f"\nTEST ACCURACY: {accuracy_test:.4f}\n")
    return {
        "SEED": SEED,
        "model": model,
        "BATCH": BATCH,
        "val_acc": round(100 * accuracy_val, 6),
        "test_acc": round(100 * accuracy_test, 6),
        "weights": best_weights
    }




if __name__ == "__main__":
    path = "/path/ladybirds/predictions_for_ga"
    folders = [p for p in os.listdir(path) if os.path.isdir(os.path.join(path, p))]
    class_cols = ['Adalia bipunctata', 'Adalia decempunctata', 'Anatis ocellata', 'Aphidecta obliterata', 'Calvia quattuordecimguttata', 'Chilocorus renipustulatus',
        'Coccinella septempunctata', 'Coccinella undecimpunctata', 'Exochomus quadripustulatus', 'Halyzia sedecimguttata', 'Harmonia axyridis', 'Harmonia quadripunctata',
        'Hippodamia variegata', 'Propylea quattuordecimpunctata', 'Psyllobora vigintiduopunctata', 'Scymnus interruptus', 'Subcoccinella vigintiquattuorpunctata', 'Tytthaspis sedecimpunctata']
    tasks = []
    seeds = list(range(42, 72))
    for BATCH in range(1, 6):
        for model in folders:
            for SEED in seeds:
                tasks.append((SEED, model, BATCH))
    results = []
    max_workers = max(1, os.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_task, SEED, model, BATCH, class_cols)
            for SEED, model, BATCH in tasks
        ]

        for future in as_completed(futures):
            results.append(future.result())

    # =========================
    # SAVE CONSOLIDATED RESULTS
    # =========================

    for BATCH in range(1, 6):
        for model in folders:

            subset = sorted([r for r in results if r["BATCH"] == BATCH and r["model"] == model], key=lambda x: x["SEED"])

            df_results = pd.DataFrame(
                {1: [r["test_acc"] for r in subset]},
                index=[r["SEED"] for r in subset]
            )

            df_val = pd.DataFrame(
                {1: [r["val_acc"] for r in subset]},
                index=[r["SEED"] for r in subset]
            )

            df_weights = pd.DataFrame(
                [r["weights"] for r in subset],
                columns=["weight"]
            )

            base_path = f'/path/ladybirds/ga_outputs/ga_1_weight_200/{model}/Batch{BATCH}'

            os.makedirs(base_path, exist_ok=True)

            df_results.to_csv(f'{base_path}/test_results_{model}.csv', index_label="RUN")
            df_val.to_csv(f'{base_path}/val_{model}.csv', index_label="RUN")
            df_weights.to_csv(f'{base_path}/WEiGHTS_{model}.csv', index=False)

    print("Processing completed successfully")