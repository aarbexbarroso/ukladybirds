import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import random
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# Função para carregar dados
def load_data(file_path):
    """
    Carrega os dados do arquivo CSV e separa em ID, rótulos reais e previsões de ambos os modelos.
    """
    data = pd.read_csv(file_path)
    ids = data.iloc[:, 0].values  # ID dos registros
    y_true = data.iloc[:, 1].values  # Saídas reais
    model_1_probs = data.iloc[:, 2:20].values  # Probabilidades do Modelo 1
    model_2_probs = data.iloc[:, 20:38].values  # Probabilidades do Modelo 2
    return ids, y_true, model_1_probs, model_2_probs

# Função de avaliação
def evaluate(y_true, combined_probs):
    """
    Avalia a precisão combinando as previsões e calculando a acurácia.
    """
    y_pred = np.argmax(combined_probs, axis=1)
    return accuracy_score(y_true, y_pred)

# Combinação das probabilidades dos modelos
def combine_probabilities(model_1_probs, model_2_probs, weight):
    """
    Combina as probabilidades dos dois modelos usando pesos específicos para cada classe.
    """
    return weight * model_1_probs + (1 - weight) * model_2_probs

# Função para criar a população inicial
def create_population(size):
    """
    Cria uma população inicial de pesos aleatórios para cada classe.
    """
    return np.random.rand(size)

# Seleção dos melhores indivíduos
def select(population, scores, num_parents):
    """
    Seleciona os melhores indivíduos da população com base na pontuação.
    """
    parents = []
    pop_size = len(population)
    
    for _ in range(num_parents):
        id1, id2 = np.random.randint(0, pop_size, size = 2)
        if scores[id1] >= scores[id2]:
            parents.append(population[id1])
        else:
            parents.append(population[id2])
    return np.array(parents)
    
# Seleção dos melhores indivíduos
def reduce_pop(population, scores, num_parents):
    """
    Seleciona os melhores indivíduos da população com base na pontuação.
    """
    selected_indices = np.argsort(scores)[-num_parents:]
    return population[selected_indices]


# Função de cruzamento
def crossover(parents, offspring_size):
    """
    Realiza o cruzamento entre os pais para gerar novos indivíduos.
    """
    offspring = []
    for _ in range(offspring_size):
        parent1, parent2 = random.sample(list(parents), 2)
        child = (parent1 + parent2) / 2
        offspring.append(child)
    return np.array(offspring)

# Função de mutação
def mutate(offspring, mutation_rate=0.1):
    """
    Aplica mutação nos indivíduos da população.
    """
    for i in range(len(offspring)):
        if random.random() < mutation_rate:
            offspring[i] += np.random.normal(0, 0.1)  # Pequena alteração
            offspring[i] = np.clip(offspring[i], 0, 1)  # Mantém o peso entre 0 e 1
    return offspring

# Algoritmo genético principal
def genetic_algorithm(y_true, model_1_probs, model_2_probs, num_generations=50, population_size=100, num_parents=100, mutation_rate=0.5):
    """
    Executa o algoritmo genético para encontrar os melhores pesos por classe.
    """
    
    population = create_population(population_size)  # População inicial
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
        
        # Cruzamento e mutação para gerar nova população
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
                
        # Atualiza a população
        population = np.concatenate((population, offspring))
        
        
        scores = []
        for weight in population:
            combined_probs = combine_probabilities(model_1_probs, model_2_probs, weight)
            accuracy = evaluate(y_true, combined_probs)
            scores.append(accuracy)
        
        # Seleção dos melhores
        scores = np.array(scores)
        
        # Melhor indivíduo desta geração
        if (generation == 0):
            best_score = max(scores)
            best_ind = np.argmax(scores)
            best_weight = population[best_ind]
        elif (max(scores)> best_score):
            best_score = max(scores)
            best_ind = np.argmax(scores)
            best_weight = population[best_ind]
        print(f"Geração {generation + 1} - Melhor precisão: {best_score:.4f}")
        #acc_plot.append(best_score)
    
    # Melhor conjunto de pesos encontrado
    print(f"Geração {generation + 1} - Melhor precisão: {best_score:.4f}")
    return best_weight

def run_task(SEED, model, BATCH, class_cols):
	# Python built-in
	random.seed(SEED)
	# NumPy
	np.random.seed(SEED)
	print(f' Iniciando SEED {SEED} para modelo {model}')
	# Carregar os dados
	file_path = f'/path/joaninhas/predictions_for_ga/{model}/Batch_{BATCH}/val_predictions.csv'  # Substitua pelo caminho do seu arquivo
	ids, y_true, model_1_probs, model_2_probs = load_data(file_path)
	# Executar o algoritmo genético
	melhores_pesos = genetic_algorithm(y_true, model_1_probs, model_2_probs)
	print(f"Melhores pesos encontrados para cada classe: {melhores_pesos}")
	combined_probs = combine_probabilities(model_1_probs, model_2_probs, melhores_pesos)
	accuracy_val = evaluate(y_true, combined_probs)
	###########################################################################################################
	# Carregar os dados
	file_path = f'/path/joaninhas/predictions_for_ga/{model}/Batch_{BATCH}/test_predictions.csv'   # Substitua pelo caminho do seu arquivo
	ids, y_true, model_1_probs, model_2_probs = load_data(file_path)
	combined_probs = combine_probabilities(model_1_probs, model_2_probs, melhores_pesos)
	accuracy_test = evaluate(y_true, combined_probs)
	df_final = pd.DataFrame(combined_probs, columns=class_cols)
	df_final.insert(0, "image_id", ids)
	df_final["true_label"] = y_true
	output_csv = f'/path/joaninhas/ga_outputs/ga_1_weight_200/{model}/Batch{BATCH}/predictions/SEED{SEED}_predictions.csv'
	os.makedirs(os.path.dirname(output_csv), exist_ok=True)
	df_final.to_csv(output_csv, index=False)
	print(f"\nPrecisão de TESTE: {accuracy_test:.4f}\n")
	return {
        "SEED": SEED,
        "model": model,
        "BATCH": BATCH,
        "val_acc": round(100 * accuracy_val, 6),
        "test_acc": round(100 * accuracy_test, 6),
        "weights": melhores_pesos
    }




if __name__ == "__main__":
    caminho = "/path/joaninhas/predictions_for_ga"
    pastas = [p for p in os.listdir(caminho) if os.path.isdir(os.path.join(caminho, p))]
    class_cols = ['Adalia bipunctata', 'Adalia decempunctata', 'Anatis ocellata', 'Aphidecta obliterata', 'Calvia quattuordecimguttata', 'Chilocorus renipustulatus',
        'Coccinella septempunctata', 'Coccinella undecimpunctata', 'Exochomus quadripustulatus', 'Halyzia sedecimguttata', 'Harmonia axyridis', 'Harmonia quadripunctata',
        'Hippodamia variegata', 'Propylea quattuordecimpunctata', 'Psyllobora vigintiduopunctata', 'Scymnus interruptus', 'Subcoccinella vigintiquattuorpunctata', 'Tytthaspis sedecimpunctata']
    tasks = []
    seeds = list(range(42, 72))
    for BATCH in range(1, 6):
        for model in pastas:
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
    # SALVAR RESULTADOS CONSOLIDADOS
    # =========================

    for BATCH in range(1, 6):
        for model in pastas:

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

            base_path = f'/path/joaninhas/ga_outputs/ga_1_weight_200/{model}/Batch{BATCH}'

            os.makedirs(base_path, exist_ok=True)

            df_results.to_csv(f'{base_path}/test_results_{model}.csv', index_label="RUN")
            df_val.to_csv(f'{base_path}/val_{model}.csv', index_label="RUN")
            df_weights.to_csv(f'{base_path}/PESOS_{model}.csv', index=False)

    print("Processamento finalizado com sucesso")


