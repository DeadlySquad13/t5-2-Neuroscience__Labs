import numpy as np
from collections import Counter

Possibilities = list[int]
Matrix = list[Possibilities]
TransitionMatrix = Matrix


# from lab1.keymappings_loop import KeymappingsLoop
# Получаем значение, соответствующее отрезку.
# v - случайная величина. Если нужно сгенерировать распределение по `p`, нужно
# подать runif(1)
# X - величины, соответствующие отрезкам.
# За отрезки отвечает параметр p, хранящий последовательность длин.
def get_value_in_range(
    v: int,
    X: Matrix,
    p: Possibilities
):
    border = 0

    for i, p_i in enumerate(p):
        border = border + p_i
        if v < border:
            return(X[i])


def make_step(current_state: int, transition_matrix: TransitionMatrix):
    probabilities_for_current_state = transition_matrix[current_state]

    possible_states = list(range(len(transition_matrix)))

    return get_value_in_range(
        np.random.uniform(),
        possible_states,
        probabilities_for_current_state
    )


def walk(initial_state, times, transition_matrix):
    if times == 0:
        return(initial_state)

    next_state = make_step(initial_state, transition_matrix)

    return walk(next_state, times - 1, transition_matrix)


NUMBER_OF_EXPERIMENTS = 1_000


def simulate_walks(
    initial_state: int,
    times_per_walk: int,
    transition_matrix: TransitionMatrix,
    number_of_experiments=NUMBER_OF_EXPERIMENTS,
):
    return [walk(initial_state, times_per_walk, transition_matrix)
            for _ in range(number_of_experiments)]


def get_results():
    transition_matrix = [
        [0.3, 0.3, 0.4],
        [0.5, 0.3, 0.2],
        [0.7, 0.1, 0.2],
    ]
    initial_state = 0
    results = simulate_walks(initial_state, 2, transition_matrix)

    return Counter(results)


def main():
    print(get_results())


if __name__ == '__main__':
    main()
