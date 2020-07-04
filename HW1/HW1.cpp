#include <bits/stdc++.h>
using namespace std;

typedef pair<int, int> Point;
typedef vector<Point> Point_vec;
typedef set<Point> Point_set;

const int INF = 1e9;
const int FOUND = -1;
int board_size;
Point_vec optimal_path;

// custom hash function for unordered_map
struct hash_pair {
    template <class T1, class T2> 
    size_t operator()(const pair<T1, T2>& p) const {
        auto hash1 = hash<T1>{}(p.first);
        auto hash2 = hash<T2>{}(p.second);
        return hash1 ^ hash2;
    }
};


// return the coordinates of next 8 steps
// move policy: 
//     (1, 2), (-1, 2), (1, -2), (-1, -2), 
//     (2, 1), (-2, 1), (2, -1), (-2, -1)
Point_vec get_next_steps(Point &pos) {
    int r, c;
    int new_r, new_c;
    r = pos.first;
    c = pos.second;
    Point_vec next_steps;
    for(int dr=-1; dr<=1; dr+=2) {
        for(int dc=-2; dc<=2; dc+=4) {
            new_r = r + dr;
            new_c = c + dc;
            if(!(new_r<0 || new_r>=board_size || new_c<0 || new_c>=board_size)) {
                next_steps.push_back(make_pair(new_r, new_c));
            }
        }
    }
    for(int dr=-2; dr<=2; dr+=4) {
        for(int dc=-1; dc<=1; dc+=2) {
            new_r = r + dr;
            new_c = c + dc;
            if(!(new_r<0 || new_r>=board_size || new_c<0 || new_c>=board_size)) {
                next_steps.push_back(make_pair(new_r, new_c));
            }
        }
    }
    return next_steps;
}


// check if the elem exists in vec
bool is_in(Point_vec &vec, Point &elem) {
    for(int i=0; i<vec.size(); i++) {
        if(vec[i] == elem) return true;
    }
    return false;
}


// compute Manhattan distance
int get_manhattan_dis(Point &p1, Point &p2) {
    return abs(p1.first-p2.first)+abs(p1.second-p2.second);
}


// heuristic function used in this assignment
int get_h_score(Point &p1, Point &p2) {
    return (int)floor(get_manhattan_dis(p1, p2) / 3.0);
}


int bfs(Point start, Point goal) {
    Point_vec path, next_steps;
    Point_set explored_set;
    queue<Point_vec> q; // implement the bfs with queue
    Point node, pos;
    int expanded = 0;

    optimal_path = Point_vec{}; // initialize the optimal path to empty vector
    q.push(Point_vec{start}); // push the starting point into the queue

    // while the queue is not empty
    while(q.size()) {
        path = q.front(); // get the front element of the queue
        q.pop();
        node = path[path.size()-1]; // current node is the last node in the path

        // if the path is found
        if(node == goal) {
            optimal_path = path;
            return expanded;
        }

        next_steps = get_next_steps(node); // get the next 8 steps
        explored_set.insert(node); // current node is expanded, add it to the explored set
        expanded++;
        for(int i=0; i<next_steps.size(); i++) {
            pos = next_steps[i];

            // push this position into queue if it is not yet expanded
            if(explored_set.find(pos) == explored_set.end()) {
                Point_vec new_path(path);
                new_path.push_back(pos);
                q.push(new_path);
            }
        }
    }
    return expanded;
}


int dfs(Point start, Point goal) {
    stack<pair<Point_vec, Point_set>> s; // implement the dfs with stack
    pair<Point_vec, Point_set> state;
    pair<int, Point> candidate;
    Point_vec path, next_steps;
    Point_set explored_set;
    Point node, pos;
    int minimum_step = 1e5;
    int expanded = 0;
    
    optimal_path = Point_vec{}; // initialize the optimal path to empty vector
    s.push(make_pair(Point_vec{start}, Point_set{})); // push the starting point and corresponding status set into the queue

    // while the stack is not empty
    while(s.size()) {
        state = s.top(); // get the top element of the stack
        s.pop();
        path = state.first;
        explored_set = state.second;
        node = path[path.size()-1]; // current node is the last node in the path

        // if the path is found
        if(node == goal) {
            // update the optimal path if the current path is better than that
            if(path.size() < minimum_step) {
                minimum_step = path.size();
                optimal_path = path;
            }
            continue; // keep going until all the path are found
        }

        // if the current length of path is longer than the one of optimal path, 
        // it is impossible to become a new optimal path
        if(path.size() > minimum_step) continue;

        next_steps = get_next_steps(node); // get the next 8 steps
        explored_set.insert(node); // current node is expanded, add it to the explored set
        expanded++;
        for(int i=0; i<next_steps.size(); i++) {
            pos = next_steps[i];

            // push this position into stack if it is not yet expanded
            if(explored_set.find(pos) == explored_set.end()) {
                Point_vec new_path(path);
                new_path.push_back(pos);
                s.push(make_pair(new_path, explored_set));
            }
        }
    }
    return expanded;
}


// DFS with limited depth
pair<Point_vec, int> deep_limited_search(Point start, Point goal, int depth) {
    stack<tuple<Point_vec, Point_set, int>> s; // implement the dfs with stack
    tuple<Point_vec, Point_set, int> state;
    pair<int, Point> candidate;
    Point_vec path, next_steps;
    Point_set explored_set;
    Point node, pos;
    int cur_depth;
    int expanded = 0;

    s.push(make_tuple(Point_vec{start}, Point_set{}, depth)); // push the starting point and corresponding status into the queue

    // while the stack is not empty
    while(s.size()) {
        state = s.top(); // get the top element of the stack
        s.pop();
        path = get<0>(state);
        explored_set = get<1>(state);
        cur_depth = get<2>(state);
        node = path[path.size()-1]; // current node is the last node in the path

        // if the path is found
        if(node == goal) return make_pair(path, expanded);

        // abandon this state if it reaches the depth limit
        if(cur_depth <= 0) continue;
        
        next_steps = get_next_steps(node); // get the next 8 steps
        explored_set.insert(node); // current node is expanded, add it to the explored set
        expanded++;
        for(int i=0; i<next_steps.size(); i++) {
            pos = next_steps[i];

            // push this position into stack if it is not yet expanded
            if(explored_set.find(pos) == explored_set.end()) {
                Point_vec new_path(path);
                new_path.push_back(pos);
                s.push(make_tuple(new_path, explored_set, cur_depth-1));
            }
        }
    }
    return make_pair(Point_vec{}, expanded);
}


int ids(Point start, Point goal) {
    pair<Point_vec, int> result;
    Point_vec path;
    int expanded = 0;
    optimal_path = Point_vec{}; // initialize the optimal path to empty vector

    // run repeatedly with increasing depth limits until the goal is found
    for(int depth=1; depth<=board_size*board_size; depth++) {
        result = deep_limited_search(start, goal, depth);
        path = result.first;
        expanded += result.second;

        // if the path is found, its size will be greater than zero
        if(path.size()) {
            optimal_path = path;
            return expanded;
        }
    }
    return expanded;
}


int a_star(Point start, Point goal) {
    priority_queue<
        pair<int, Point_vec>,
        vector<pair<int, Point_vec>>,
        greater<pair<int, Point_vec>>> open_set; // implement the A* with min-heap
    unordered_map<Point, int, hash_pair> g_score, h_score, f_score;
    pair<int, Point_vec> state;
    Point_vec path, next_steps;
    Point_set closed_set;
    Point node, pos;
    int g, h, f;
    int expanded = 0;

    optimal_path = Point_vec{}; // initialize the optimal path to empty vector
    g_score[start] = 0; // initial g-score of starting point is 0
    h_score[start] = get_h_score(start, goal); // initial h-score of starting point is h(start)
    f_score[start] = g_score[start] + h_score[start];
    open_set.push(make_pair(f_score[start], Point_vec{start})); // push the starting point and corresponding status into the min-heap

    // while min-heap is not empty
    while(open_set.size()) {
        state = open_set.top(); // get the top element of the min-heap
        open_set.pop();
        path = state.second;
        node = path[path.size()-1]; // current node is the last node in the path

        // if the path is found
        if(node == goal) {
            optimal_path = path;
            return expanded;
        }

        next_steps = get_next_steps(node); // get the next 8 steps
        closed_set.insert(node); // current node is expanded, add it to the closed set
        expanded++;
        for(int i=0; i<next_steps.size(); i++) {
            pos = next_steps[i];

            // consider this position if it is not yet expanded
            if(closed_set.find(pos) == closed_set.end()) {
                g = g_score[node] + 1;
                h = get_h_score(pos, goal);
                f = g + h; // update the g-score and compute h(pos)

                // push this position into min-heap if
                //     1. this position is not yet visited
                //     2. current g-score if smaller than the old one
                //        (it means that this position can be reached with less costs)
                if(g_score.find(pos)==g_score.end() || g<g_score[pos]) {
                    g_score[pos] = g;
                    h_score[pos] = h;
                    f_score[pos] = f;
                    Point_vec new_path(path);
                    new_path.push_back(pos);
                    open_set.push(make_pair(f, new_path));
                }
            }
        }
    }
    return expanded;
}


pair<int, int> ida_star_search(Point_vec path, Point goal, int g, int thres) {
    pair<int, int> result;
    Point_vec next_steps;
    Point node, pos;
    int min, f, t;
    int expanded = 0;

    node = path[path.size()-1]; // current node is the last node in the path
    f = g + get_h_score(node, goal); // compute the f-score from current node to goal
    
    // update the threshold if f-score is greater than it
    if(f > thres) return make_pair(f, expanded);

    // if the path is found
    if(node == goal) {
        optimal_path = path;
        return make_pair(FOUND, expanded);
    }

    min = INF;
    next_steps = get_next_steps(node); // get the next 8 steps
    expanded++;

    for(int i=0; i<next_steps.size(); i++) {
        pos = next_steps[i];

        // if ths position is not in current path
        if(!is_in(path, pos)) {
            Point_vec new_path(path);
            new_path.push_back(pos);
            result = ida_star_search(new_path, goal, g+1, thres); // go to next step and see if it can find the path or update the threshold
            t = result.first;
            expanded += result.second;

            // if the path id found
            if(t == FOUND) return make_pair(t, expanded);

            // update the threshold
            if(t < min) min = t;
        }
    }
    return make_pair(min, expanded);
}


int ida_star(Point start, Point goal) {
    int t, thres = get_h_score(start, goal);
    int expanded = 0;
    pair<int, int> result;
    Point_vec path{start};
    optimal_path = Point_vec{};
    while(1) {
        result = ida_star_search(path, goal, 0, thres);
        t = result.first; // keep going and update the threshold until the path is found
        expanded += result.second;
        if(t == FOUND) return expanded;
        if(t > INF) optimal_path = Point_vec{}; // if the path doesn't exist
        thres = t;
    }
    return expanded;
}


int solve(int search_func, int start_x, int start_y, int goal_x, int goal_y) {
    Point start = make_pair(start_x, start_y);
    Point goal = make_pair(goal_x, goal_y);
    int expanded;
    switch(search_func) {
    case 0:
        return bfs(start, goal);
        break;
    case 1:
        return dfs(start, goal);
        break;
    case 2:
        return ids(start, goal);
        break;
    case 3:
        return a_star(start, goal);
        break;
    case 4:
        return ida_star(start, goal);
        break;
    default:
        return -1;
        break;
    }
}


// write the optimal path to the file
void output_path() {
    fstream fout;
    fout.open("path", ios::out);
    for(int i=0; i<optimal_path.size(); i++) {
        fout << optimal_path[i].first << " " << optimal_path[i].second << endl;
    }
    fout.close();
}


int main(int argc, char *argv[]) {
    if(argc < 7) {
        cout << "Usage: ./HW1 <searching-method> <board-size> <start-x> <start-y> <goal-x> <goal-y>\n";
        cout << "\nSearching method:\n0) BFS\n1) DFS\n2) IDS\n3) A*\n4) IDA*\n";
    }
    else {
        int search_func = stoi(string(argv[1]));
        board_size = stoi(string(argv[2]));
        int start_x = stoi(string(argv[3]));
        int start_y = stoi(string(argv[4]));
        int goal_x = stoi(string(argv[5]));
        int goal_y = stoi(string(argv[6]));
        int expanded = 0;

        if(search_func > 4) {
            cout << "Unknown searching method.\n";
        }
        else if(start_x<0 || start_x>=board_size || goal_x<0 || goal_x>=board_size) {
            cout << "Constraint:\n0 <= x, y <= " << board_size << endl;
        }
        else {
            vector<string> algo{"BFS", "DFS", "IDS", "A*", "IDA*"};
            expanded = solve(search_func, start_x, start_y, goal_x, goal_y);
            cout << algo[search_func] << endl;
            cout << "Expanded: " << expanded << endl;
            cout << "Steps: " << optimal_path.size()-1 << endl;
            cout << "--------------------\n";
            output_path();
        }
    }
    return 0;
}
