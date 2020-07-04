import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from collections import OrderedDict

MINE = 100
SAFE = 200
ASSIGN = 0
UNASSIGN = 1


# plot the board
# variables and safe places will be marked as blue cell
# mines will be marked as red cell
def plot_board(board, fig, subplot):
    ax = fig.add_subplot(subplot)
    board_size = board.shape[0]
    args = OrderedDict(
        fontsize=20,
        horizontalalignment='center',
        verticalalignment='center')

    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i, j]==-1 or board[i, j]==SAFE:
                ax.fill([j, j, j+1, j+1], [board_size-i-1, board_size-i, board_size-i, board_size-i-1], c='b', alpha=0.3)
            elif board[i, j] == MINE:
                ax.fill([j, j, j+1, j+1], [board_size-i-1, board_size-i, board_size-i, board_size-i-1], c='r', alpha=0.3)
            else:
                ax.annotate(board[i, j], (j+0.5, board_size-i-0.6), **args)

    ax.set_xticks(np.arange(board_size+1))
    ax.set_yticks(np.arange(board_size+1))
    ax.set_xlim([0, board_size])
    ax.set_ylim([0, board_size])
    ax.tick_params(bottom=False, top=False, left=False, right=False)
    ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    ax.grid(b=True, which='major', color='k', linestyle='-')
    ax.set_aspect('equal')


# generate a board randomly with 10 mines and 16 local constraints
def generate_board(board_size=6, mine_num=10, cstr_num=16):
    mines = set()
    while len(mines) < mine_num:
        i, j = np.random.randint(0, board_size, 2)
        mines.add((i, j))
    board = np.zeros((board_size+2, board_size+2), dtype=np.int32)
    for mine in mines:
        board[mine[0]+1, mine[1]+1] = -1
        i, j = mine[0]+1, mine[1]+1
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                new_i = i + di
                new_j = j + dj
                if board[new_i, new_j] == -1:
                    continue
                board[new_i, new_j] += 1
    board = board[1:-1, 1:-1]
    i, j = np.where(board>=0)
    hints = np.asarray(list(zip(i, j)))
    safe_num = board_size**2 - mine_num
    mask_num = safe_num - cstr_num
    mask_idx = np.random.choice(np.arange(safe_num), mask_num, replace=False)
    mask_hints = hints[mask_idx]
    for i, j in mask_hints:
        board[i, j] = -1
    board_str = ' '.join(board.ravel().astype(str))
    board_str = f'{board_size} {board_size} {mine_num} {board_str}'
    return board_str


# find the neighbor cells of given position, hint cells will be ignored
def get_neighbors(board, i, j):
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di==0 and dj==0:
                continue
            new_i = i + di
            new_j = j + dj
            if (new_i<0 or new_i>=board.shape[0] or
                    new_j<0 or new_j>=board.shape[1]):
                continue

            # ignore it if it is a hint cell
            if board[new_i, new_j] != -1:
                continue
            neighbors.append((i+di, j+dj))
    return neighbors


# do the forward checking on given status
def forward_checking(board, mine_num, cstr, assign, unassign):
    while True:
        cur_mine_num = sum([v for k, v in assign.items()])
        # if current number of mine cells exceeds total number of mines,
        # it is impossible to derive an answer from this status
        if cur_mine_num > mine_num:
            return False

        # compute the upper bound and lower bound of the sum of domains of unassigned variables
        # if upper bound is smaller than number of mines remaining or
        # lower bound is larger than number of mines remaining, 
        # it is impossible to derive an answer from this status
        remain_mine_num = mine_num - cur_mine_num
        upper_bound = sum([max(v) for k, v in unassign.items()])
        lower_bound = sum([min(v) for k, v in unassign.items()])
        if upper_bound < remain_mine_num:
            return False
        if lower_bound > remain_mine_num:
            return False

        prev_assign = dict(assign)
        prev_unassign = dict(unassign)
        for ((i, j), hint) in cstr:
            lower_bound = 0
            upper_bound = 0
            neighbors = get_neighbors(board, i, j)
            for pos in neighbors:
                if pos in assign:
                    hint -= assign[pos]
                else:
                    upper_bound += max(unassign[pos])
                    lower_bound += min(unassign[pos])

            # if the lower bound is larger than the hint, the constraint cannot be satisfied.
            if lower_bound > hint:
                return False

            # if the upper bound is smaller than the hint, the constraint cannot be satisfied.
            if upper_bound < hint:
                return False

            # if the lower bound equals the hint, the domains of all the
            # unassigned variables in the constraint should be limited to
            # their respective minimal values.
            if lower_bound == hint:
                for pos in neighbors:
                    if pos not in assign:
                        unassign[pos] = [min(unassign[pos])]

            # if the upper bound equals the hint, the domains of all the
            # unassigned variables in the constraint should be limited to
            # their respective maximal values
            if upper_bound == hint:
                for pos in neighbors:
                    if pos not in assign:
                        unassign[pos] = [max(unassign[pos])]

        # keep doing forward checking until the domains of all the unassigned variables remain unchanged
        if prev_assign==assign and prev_unassign==unassign:
            break
    return True


# check whether the final assignment is valid or not
def is_valid_answer(board, mine_num, cstr, assign):
    total_mine = sum([v for k, v in assign.items()])
    # global constraint
    if total_mine != mine_num:
        return False

    # contraint given by each hint
    for ((i, j), hint) in cstr:
        neighbors = get_neighbors(board, i, j)
        for n in neighbors:
            hint -= assign[n]
        if hint != 0:
            return False
    return True


# compute the degree of a given variable
# degree = number of constrains regard to this variable
def get_degree(board, i, j):
    degree = 0
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di==0 and dj==0:
                continue
            new_i = i + di
            new_j = j + dj
            if (new_i<0 or new_i>=board.shape[0] or
                    new_j<0 or new_j>=board.shape[1]):
                continue
            if board[new_i, new_j] != -1:
                degree += 1
    return degree


# find the answer using backtrack search
# use forward_checking, MRV, degree_heuristic to control the detail of backtrack search
def solve(
        board, mine_num, forword_checking=True,
        MRV=True, degree_heuristic=True):
    cstr = []

    # assign = {
    #     variable: assignment (is or is not a mine)
    # }
    assign = {}

    # unassign = {
    #     variable: domain
    # }
    unassign = {}
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i, j] == -1:
                unassign[(i, j)] = [1, 0]
            else:
                cstr.append(((i, j), board[i, j]))
    cstr = sorted(cstr, key=lambda t: t[1])
    expand = 0
    stack = []
    stack.append((assign, unassign))
    while stack:
        node = stack.pop()

        # do the forward checking if forward_checking is True
        if ((not forward_checking) or
                forward_checking(board, mine_num, cstr, node[ASSIGN], node[UNASSIGN])):

            # check the answer if there is no unassigned variables
            if not node[UNASSIGN].keys():
                if is_valid_answer(board, mine_num, cstr, node[ASSIGN]):
                    return (expand, node[ASSIGN])
                else:
                    continue

            # computer the order of child node to be pushed into the stack
            t = {}
            for pos in node[UNASSIGN]:

                # MRV heuristic
                domain = 0
                if MRV:
                    domain = len(node[UNASSIGN][pos])

                # degree heuristic
                degree = 0
                if degree_heuristic:
                    degree = get_degree(board, pos[0], pos[1])
                t[pos] = (-domain, degree)
            order = [k for k, v in sorted(t.items(), key=lambda elem: elem[1])]
            for pos in order:
                for value in node[UNASSIGN][pos]:
                    new_assign = dict(node[ASSIGN])
                    new_unassign = dict(node[UNASSIGN])

                    # assign a value to this variable
                    new_assign[pos] = value

                    # remove this variable from unassigned list
                    new_unassign.pop(pos, None)
                    stack.append((new_assign, new_unassign))
            expand += 1


if __name__ == '__main__':
    # generate a board randomly
    board = generate_board()
    board = np.asarray(board.strip().split(' ')).astype(np.int32)
    board_size = board[0:2]
    mine_num = board[2]
    board = board[3:].reshape(board_size)

    # plot the problem
    fig = plt.figure()
    plot_board(board, fig, 111)
    plt.title('Problem', fontsize=14)
    plt.savefig(f'problem', dpi=300, transparent=True)
    plt.close(fig)

    # solve the board with backtrack search
    fc, mrv, dh = True, True, True
    start = datetime.now()
    expand, assign = solve(
        board, mine_num, forword_checking=fc, MRV=mrv, degree_heuristic=dh)
    end = datetime.now()
    delta = end - start

    print(f'size {board_size[0]}x{board_size[1]}')
    print(f'{mine_num} mines\nexpand {expand} nodes')
    print(f'forward checking: {fc}')
    print(f'MRV: {mrv}')
    print(f'degree heuristic: {dh}')
    print(delta, end='\n\n')

    # fill the board with the solution
    for pos in assign:
        board[pos] = MINE if assign[pos] else SAFE

    # plot the solution
    fig = plt.figure()
    plot_board(board, fig, 111)
    plt.title('Solution', fontsize=14)
    filename = f'solution'
    if fc:
        filename = f'{filename}_fc'
    if mrv:
        filename = f'{filename}_mrv'
    if dh:
        filename = f'{filename}_dh'
    filename = f'{filename}.png'
    plt.savefig(f'{filename}', dpi=300, transparent=True)
    plt.close(fig)
