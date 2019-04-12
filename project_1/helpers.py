from enums import MazePositionType, color_map
import numpy as np
import random
import matplotlib.cm as cm
import matplotlib.pyplot as plt  

def maze_generator(): 
    maze = np.zeros((60, 60))
    
    maze[0][0] = MazePositionType.ROBOT # indicates the current position of the robot
    
    current_line = 0
    for line in maze:
        if current_line % 8 == 0 and current_line != 0:
            if bool(random.getrandbits(1)):
                line[10:] = MazePositionType.WALL
            else:
                line[:-10] = MazePositionType.WALL
        current_line = current_line + 1
        
    maze[59][59] = MazePositionType.GOAL # indicates the goal
    return maze


def does_visited_states_contain(visited_states, state):
    for s in visited_states:
        if s["x"] == state["x"] and s["y"] == state["y"] and s["angle"] == state["angle"]:
            return True
    return False

 

def plot_tile_map(tiles, show_img=True):
    ######## THIS IS FOR IMSHOW ######################################
    width, height = np.shape(tiles)
    data = tiles.astype(int)
    c = np.array([[color_map.get(v, color_map.get(MazePositionType.VISITED)) for v in row] for row in data], dtype='B')

    ######## THIS IS FOR CMAP ##########################################
    cmap = cm.get_cmap('viridis', len(color_map))
    aaaaaa = np.array([tuple(np.array(color_map[b]) / 255.) for b in MazePositionType], np.dtype('float,float,float,float'))
    cmap.colors = aaaaaa

    ######## THIS IS FOR PLOTTING ######################################
    # two subplots, plot immediately the imshow
    f, ax1 = plt.subplots(nrows=1)
    a1 = ax1.imshow(c, cmap=cmap)
    # Major ticks
    ax1.set_xticks(np.arange(width))
    ax1.set_yticks(np.arange(height))

    # Labels for major ticks
    ax1.set_xticklabels(np.arange(width))
    ax1.set_yticklabels(np.arange(height))

    # Minor ticks
    ax1.set_xticks(np.arange(width) - .5, minor=True)
    ax1.set_yticks(np.arange(height) - .5, minor=True)

    # Gridlines based on minor ticks
    line_color = (90 / 255., 90 / 255., 90 / 255.)
    ax1.grid(which='minor', color=line_color, linestyle='--', linewidth=1)

    cb = f.colorbar(a1, ax=ax1)
    cb.set_ticks(np.arange(len(color_map)) * 36)
    cb.set_ticklabels(['GOAL', 'ROBOT', 'EMPTY', 'WALL', 'VISITED', 'PATH'])
    # set ticks for all subplots
    plt.setp((ax1,), xticks=np.arange(width), yticks=np.arange(height))
    if show_img:
        plt.show()