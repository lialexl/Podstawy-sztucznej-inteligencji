import random

def fitness(board):
    n = len(board)
    attacks = 0
    for i in range(n):
        for j in range(i + 1, n):
            if board[i] == board[j] or abs(board[i] - board[j]) == abs(i - j):
                attacks += 1
    return attacks

def crossover(p1, p2):
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    c1 = p1[:a] + p2[a:b] + p1[b:]
    c2 = p2[:a] + p1[a:b] + p2[b:]
    return c1, c2

def mutate(board, rate):
    n = len(board)
    if random.random() < rate:
        i = random.randint(0, n - 1)
        board[i] = random.randint(0, n - 1)
    return board

def genetic_n_queens(n=8, pop_size=1000, mutation_rate=0.2, max_iter=10000):
    population = [[random.randint(0, n - 1) for i in range(n)] for i in range(pop_size)]
    for iteration in range(max_iter):
        population.sort(key=fitness)
        if fitness(population[0]) == 0:
            return population[0]
        new_population = population[:pop_size // 2]
        while len(new_population) < pop_size:
            p1, p2 = random.choices(population, weights=[1/(1+fitness(b)) for b in population], k=2)
            c1, c2 = crossover(p1, p2)
            new_population.append(mutate(c1, mutation_rate))
            if len(new_population) < pop_size:
                new_population.append(mutate(c2, mutation_rate))
        population = new_population
    return None

def print_board(board):
    n = len(board)
    print("   " + "".join(chr(ord('A') + i) for i in range(n)))
    for i in range(n):
        row = ['.'] * n
        row[board[i]] = 'X'
        print(f"{i+1:2d} " + "".join(row))

if __name__ == "__main__":
    N = int(input("Podaj N: "))
    if N in (2, 3):
        print("Brak rozwiązań dla N=2 i N=3.")
    else:
        result = genetic_n_queens(N)
        if result:
            print("Rozwiązanie:", result)
            print_board(result)
        else:
            print("Nie znaleziono rozwiązania.")
