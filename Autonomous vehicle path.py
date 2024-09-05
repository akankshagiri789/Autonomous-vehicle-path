import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue

# Grid size
GRID_SIZE = 10

# Define directions (up, down, left, right)
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

# Heuristic function for A* (Manhattan distance)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# A* Path Planning Algorithm
def astar(grid, start, goal):
    open_list = PriorityQueue()
    open_list.put((0, start))  # (priority, node)
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while not open_list.empty():
        current = open_list.get()[1]

        # If goal is reached, reconstruct the path
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for direction in DIRECTIONS:
            neighbor = (current[0] + direction[0], current[1] + direction[1])

            if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE:  # Stay within grid
                if grid[neighbor] == 1:  # Skip obstacle
                    continue
                
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    open_list.put((f_score[neighbor], neighbor))
    
    return None  # No path found

# Obstacle Avoidance (Simple Reactive Approach)
def avoid_obstacles(grid, path):
    safe_path = []
    for point in path:
        x, y = point
        if grid[x, y] == 0:
            safe_path.append(point)
        else:
            # Simple detour to avoid obstacle
            for direction in DIRECTIONS:
                neighbor = (x + direction[0], y + direction[1])
                if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE:
                    if grid[neighbor] == 0:
                        safe_path.append(neighbor)
                        break
    return safe_path

# Visualize the grid, path, and obstacles
def visualize(grid, path, start, goal):
    plt.imshow(grid, cmap="gray_r")
    plt.plot(start[1], start[0], "go")  # Start point (green)
    plt.plot(goal[1], goal[0], "ro")    # Goal point (red)

    if path:
        path = np.array(path)
        plt.plot(path[:, 1], path[:, 0], "b-", linewidth=2)  # Path (blue)

    plt.grid(True)
    plt.show()

# Main function
if __name__ == "__main__":
    # Create grid (0 = free, 1 = obstacle)
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    
    # Define obstacles (1 = obstacle)
    obstacles = [(3, 3), (3, 4), (3, 5), (7, 6), (7, 7)]
    for obstacle in obstacles:
        grid[obstacle] = 1

    # Start and goal positions
    start = (0, 0)
    goal = (9, 9)

    # Find path using A*
    path = astar(grid, start, goal)
    
    if path:
        print("Path found:", path)
        # Obstacle avoidance
        safe_path = avoid_obstacles(grid, path)
        visualize(grid, safe_path, start, goal)
    else:
        print("No path found.")
