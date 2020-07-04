import os
import sys
import numpy as np
import matplotlib.pyplot as plt

colors = [
    'red', 'darkorange', 'gold', 'limegreen', 'turquoise', 'dodgerblue',
    'blue'
]
search_method = ['BFS', 'DFS', 'IDS', 'A_star', 'IDA_star']


def plot_path(path, method, board_size):
    if not os.path.exists(f'./images_{board_size}'):
        os.mkdir(f'./images_{board_size}')
    # print(search_method[method].replace('_star', '*'), end=': ')
    print(f'{len(path)-1} steps')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(path)-1):
        u = path[i]
        v = path[i+1]
        ax.plot(
            [u[1], v[1]],
            [board_size-u[0]-1, board_size-v[0]-1],
            linewidth=5, c=colors[i%len(colors)],
            solid_capstyle='round', alpha=0.7)

    ax.plot(path[0][1], board_size-path[0][0]-1, marker='o', markersize=10, c='g')
    ax.plot(path[-1][1], board_size-path[-1][0]-1, marker='o', markersize=10, c='b')

    ax.set_xticks(np.arange(board_size))
    ax.set_xticklabels(np.arange(board_size))
    ax.set_yticks(np.arange(board_size))
    ax.set_yticklabels(np.arange(board_size-1, -1, -1))
    ax.set_xlim([-0.5, board_size-0.5])
    ax.set_ylim([-0.5, board_size-0.5])
    ax.tick_params(bottom=False, top=True, left=True, right=False)
    ax.tick_params(labelbottom=False, labeltop=True, labelleft=True, labelright=False)
    ax.grid(b=True, which='major', color='k', linestyle='-')
    ax.set_aspect('equal')

    title = f'{search_method[method].replace("_star", "*")}: {path[0]} -> {path[-1]}'
    title = f'{title}\n{len(path)-1} step(s)'
    plt.title(title)
    plt.tight_layout()
    filename = f'{board_size}_{path[0][0]}-{path[0][1]}_{path[-1][0]}-{path[-1][1]}.png'
    filename = f'{search_method[method]}_{filename}'
    filename = f'./images_{board_size}/{filename}'
    # plt.savefig(filename, dpi=300, transparent=True)
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage:')
        print(f'python3 {sys.argv[0]} <searching-method> <board_size>')
        print('\nSearching method:\n0) BFS\n1) DFS\n2) IDS\n3) A*\n4) IDA*')
    else:
        method = int(sys.argv[1])
        board_size = int(sys.argv[2])
        path = []
        with open('./path', 'r') as file:
            for line in file:
                if not line.strip():
                    break
                x, y = line.strip().split(' ')
                path.append((int(x), int(y)))
        plot_path(path, method, board_size)
        os.remove('./path')
