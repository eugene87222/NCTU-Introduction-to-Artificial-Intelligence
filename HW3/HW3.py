import os
import time
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations as comb
from collections import OrderedDict

MINE = 100

OPEN = 200
FLAG = 300

board_meta = {
    'easy': {
        'board_size': (9, 9),
        'mine_num': 10,
        'annot_size': 20
    },
    'medium': {
        'board_size': (16, 16),
        'mine_num': 25,
        'annot_size': 10
    },
    'hard': {
        'board_size': (16, 30),
        'mine_num': 99,
        'annot_size': 8
    }
}


# plot the board with the answer and current status
# just for visualization
def plot_board(status, ans, annot_size):
    l, w = status.shape
    args_mine = OrderedDict(
        color='r',
        fontsize=annot_size,
        horizontalalignment='center',
        verticalalignment='center')
    args_hint = OrderedDict(
        fontsize=annot_size,
        horizontalalignment='center',
        verticalalignment='center')

    for i in range(l):
        for j in range(w):
            if status[i, j] == -1:
                plt.fill([j, j, j+1, j+1], [l-i-1, l-i, l-i, l-i-1], c='gray', alpha=0.3)
            elif status[i, j] == FLAG:
                plt.fill([j, j, j+1, j+1], [l-i-1, l-i, l-i, l-i-1], c='yellow', alpha=0.3)
            elif status[i, j] == OPEN:
                plt.fill([j, j, j+1, j+1], [l-i-1, l-i, l-i, l-i-1], c='lime', alpha=0.3)
            if ans[i, j] == MINE:
                plt.annotate('x', (j+0.5, l-i-0.5), **args_mine)
            else:
                plt.annotate(ans[i, j], (j+0.5, l-i-0.6), **args_hint)

    plt.gca().set_xticks(np.arange(w+1))
    plt.gca().set_yticks(np.arange(l+1))
    plt.gca().set_xlim([0, w])
    plt.gca().set_ylim([0, l])
    plt.gca().tick_params(bottom=False, top=False, left=False, right=False)
    plt.gca().tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    plt.gca().grid(b=True, which='major', color='k', linestyle='-')
    plt.gca().set_aspect('equal')


# generate a board with given size and number of mines randomly
# return the answer and initial safe cells
def generate_board(board_size, mine_num, init_safe_cell_num):
    mines = set()
    while len(mines) < mine_num:
        i = np.random.randint(0, board_size[0])
        j = np.random.randint(0, board_size[1])
        mines.add((i, j))
    ans = np.zeros((board_size[0]+2, board_size[1]+2), dtype=np.int32)
    for mine in mines:
        ans[mine[0]+1, mine[1]+1] = MINE
        i, j = mine[0]+1, mine[1]+1
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                new_i = i + di
                new_j = j + dj
                if ans[new_i, new_j] == MINE:
                    continue
                ans[new_i, new_j] += 1
    ans = ans[1:-1, 1:-1]
    safe_cell = np.argwhere(ans!=MINE)
    idx = np.random.choice(np.arange(0, safe_cell.shape[0]), init_safe_cell_num, replace=False)

    return ans, safe_cell[idx]


# get the eight neighbors of given cell
def get_neighbors(board_size, i, j):
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di==0 and dj==0:
                continue
            new_i = i + di
            new_j = j + dj
            if (new_i<0 or new_i>=board_size[0] or
                    new_j<0 or new_j>=board_size[1]):
                continue
            neighbors.append(f'{new_i},{new_j}')
    return neighbors


# get all unmarked cells
def get_unmark_cells(board_size, KB0):
    cells = []
    for i in range(board_size[0]):
        for j in range(board_size[1]):
            if f'{i},{j}' not in KB0:
                cells.append(f'{i},{j}')
    return cells


# check whether the two clauses are identical
def duplicate_pairwise(clause1, clause2):
    match = 0
    for literal in clause1:
        if literal in clause2:
            match += 1
    if match==len(clause2) and len(clause1)==len(clause2):
        return True
    return False


# check whether there exists any clause identical to the given clause in the
# knowledge base 
def duplicate(KB, clause):
    for sentence in KB:
        if duplicate_pairwise(clause, sentence):
            return True
    return False


# subsumption for single-literal clause
def subsumption_single(KB, clause):
    for i in range(len(KB)):
        if clause in KB[i]:
            KB[i] = []


# subsumption between two clauses
def subsumption_pairwise(KB, i, j):
    clause1, clause2 = KB[i], KB[j]
    match = 0
    for literal in clause1:
        if literal in clause2:
            match += 1
    if match==len(clause1) and len(clause1)<len(clause2):
        KB[j] = []
        return True
    elif match==len(clause2) and len(clause2)<len(clause1):
        KB[i] = []
        return True
    return False


# subsumption for given clause to the knowledge base
def subsumption(KB, clause):
    insert = True
    update = False
    for i in range(len(KB)):
        sentence = KB[i]
        if not len(sentence):
            continue
        match = 0
        for literal in clause:
            if literal in sentence:
                match += 1
        if match==len(sentence) and len(sentence)<=len(clause):
            insert = False
        elif match==len(clause) and len(sentence)>len(clause):
            KB[i] = []
            update = True
    if insert:
        KB.append(clause)
        update = True
    return update


# generate new clause if there is only one pair on complementary literals
# between two clauses
def comp(clause1, clause2):
    c1 = list(clause1)
    c2 = list(clause2)
    match = 0
    for literal in clause1:
        if 'not' in literal:
            l = literal[4:]
        else:
            l = f'not {literal}'
        if l in clause2:
            c1.remove(literal)
            c2.remove(l)
            match += 1
    if match == 1:
        return list(set(c1+c2))
    else:
        return []


if __name__ == '__main__':
    level = 'hard'
    meta = board_meta[level]
    board_size = meta['board_size']
    mine_num = meta['mine_num']
    annot_size = meta['annot_size']
    cell_num = board_size[0] * board_size[1]
    unmark_mine_num = mine_num
    unmark_cell_num = cell_num
    init_safe_cell_num = int(np.round(np.sqrt(cell_num)))
    ans, init_safe_cell = generate_board(board_size, mine_num, init_safe_cell_num)

    folder = f'{level}_{init_safe_cell_num}_{int(time.time())}'
    os.mkdir(folder)

    KB0 = {}
    KB = []
    # add initial safe cells into knowledge base
    for cell in init_safe_cell:
        clause = ','.join(cell.astype(str))
        clause = f'not {clause}'
        KB.append([clause])

    print(init_safe_cell)
    # initialize the status
    # -1 for unmarked
    # OPEN for marked as safe
    # FLAG for marked as mine
    status = np.ones(ans.shape, dtype=np.int32) * -1
    cnt, not_found = 0, 0
    # while knowledge base is not empty
    while KB:
        # if the number of unmarked cells is less than 10, add the global
        # constaints to knowledge base
        if unmark_cell_num <= 10:
            unmark_cells = get_unmark_cells(board_size, KB0)
            for clause in list(comb(unmark_cells, unmark_cell_num-unmark_mine_num+1)):
                clause = list(clause)
                if not duplicate(KB, clause):
                    subsumption(KB, clause)
            for clause in list(comb(unmark_cells, unmark_mine_num+1)):
                clause = list(clause)
                for i in range(len(clause)):
                    clause[i] = f'not {clause[i]}'
                if not duplicate(KB, clause):
                    subsumption(KB, clause)
        KB = sorted(KB, key=lambda t: len(t))
        found = False
        for i in range(len(KB)):
            clause = KB[i]
            # look for a single-literal clause
            if len(clause) == 1:
                found = True
                print(cnt, clause[0])
                # this cell is safe, update the status (OPEN)
                if clause[0].startswith('not'):
                    # put the marked cell into KB0
                    KB0[clause[0][4:]] = False
                    # remove that literal from knowledge base
                    KB[i] = []
                    unmark_cell_num -= 1
                    subsumption_single(KB, clause[0])
                    for j in range(len(KB)):
                        if clause[0][4:] in KB[j]:
                            KB[j].remove(clause[0][4:])
                    cell = clause[0][4:].split(',')
                    r, c = int(cell[0]), int(cell[1])
                    status[r, c] = OPEN
                    hint = ans[r, c]
                    neighbors = get_neighbors(board_size, r, c)
                    for j in range(len(neighbors)):
                        # only consider the unmarked cells
                        if neighbors[j] in KB0:
                            if KB0[neighbors[j]] == True:
                                hint -= 1
                            neighbors[j] = ''
                    neighbors = [n for n in neighbors if len(n) > 0]
                    # m = number of unmarked neighbors
                    # n = hint
                    # (m == n): insert the m single-literal positive clauses
                    # to the knowledge base, one for each unmarked neighbor
                    if hint == len(neighbors):
                        for n in neighbors:
                            if not duplicate(KB, n):
                                KB.append([n])
                    # (n == 0): insert the m single-literal negative clauses
                    # to the knowledge base, one for each unmarked neighbor
                    elif hint == 0:
                        for n in neighbors:
                            if not duplicate(KB, f'not {n}'):
                                KB.append([f'not {n}'])
                    # (m > n > 0): generate CNF clauses and add them to the
                    # knowledge base
                    elif len(neighbors) > hint:
                        for clause in list(comb(neighbors, len(neighbors)-hint+1)):
                            clause = list(clause)
                            if not duplicate(KB, clause):
                                subsumption(KB, clause)
                        for clause in list(comb(neighbors, hint+1)):
                            clause = list(clause)
                            for i in range(len(clause)):
                                clause[i] = f'not {clause[i]}'
                            if not duplicate(KB, clause):
                                subsumption(KB, clause)
                    else:
                        print(f'ERROR: hint: {hint}, neighbors: {len(neighbors)}')
                # this cell is mine, update the status (FLAG)
                else:
                    # put the marked cell into KB0
                    KB0[clause[0]] = True
                    # remove that literal from knowledge base
                    KB[i] = []
                    unmark_cell_num -= 1
                    unmark_mine_num -= 1
                    subsumption_single(KB, clause[0])
                    for j in range(len(KB)):
                        if f'not {clause[0]}' in KB[j]:
                            KB[j].remove(f'not {clause[0]}')
                    cell = clause[0].split(',')
                    r, c = int(cell[0]), int(cell[1])
                    status[r, c] = FLAG
                # plot the current status
                # plot_board(status, ans, annot_size)
                # plt.savefig(f'{folder}/{cnt}.png', dpi=300, transparent=True)
                # plt.clf()
                cnt += 1
                break
        # if we cannot found a single-literal clause
        if not found:
            print('Pairwise matching')
            update = False
            not_found += 1
            print(f'size of KB: {len(KB)}')
            for i in range(len(KB)):
                for j in range(i+1, len(KB)):
                    if len(KB[i])==0 or len(KB[j])==0:
                        continue
                    # check whether the two clauses are identical
                    if not duplicate_pairwise(KB[i], KB[j]):
                        if not subsumption_pairwise(KB, i, j):
                            if len(KB[i])>2 and len(KB[j])>2:
                                continue
                            new_clause = comp(KB[i], KB[j])
                            if new_clause:
                                if not duplicate(KB, new_clause):
                                    if subsumption(KB, new_clause):
                                        update = True
                        else:
                            update = True
                    else:
                        KB[j] = []
                        update = True
            # stop if the knowledge base didn't update after pairwise matching
            # (can neither find a single-literal clause nor generate any new
            # clause from the knowkedge base)
            if not update:
                break
        KB = [t for t in KB if len(t) > 0]
    print('--------------------')
    print(f'{cnt} cells marked')
    print(f'Pairwise matching: {not_found} times')
    os.rename(folder, f'{folder} {cnt} {not_found}')
