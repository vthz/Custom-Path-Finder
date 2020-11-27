# Shaheen Naiyer -15102019 "Hey there!!!"


import cv2
import numpy as np
import pandas as pd

drawing = False
mode = True
startx, starty = -1, -1
endx, endy = -1, -1
chances = 2
listt = []


class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


# This function return the path of the search
def return_path(current_node, maze):
    path = []
    no_rows, no_columns = np.shape(maze)
    # here we create the initialized result maze with -1 in every position
    result = [[0 for i in range(no_columns)] for j in range(no_rows)]
    current = current_node
    while current is not None:
        cx, cy = map(int, current.position)
        path.append(current.position)
        current = current.parent
        cv2.circle(img, (cy, cx), 1, (0, 0, 255), -1)
    # Return reversed path as we need to show from start to end path
    path = path[::-1]
    start_value = 10000
    # we update the path of start to end found by A-star serch with every step incremented by 1
    for i in range(len(path)):
        result[path[i][0]][path[i][1]] = start_value
        # start_value += 1
    return result


def search(maze, cost, start, end):

    # Create start and end node with initized values for g, h and f
    start_node = Node(None, tuple(start))
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, tuple(end))
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both yet_to_visit and visited list
    # in this list we will put all node that are yet_to_visit for exploration.
    # From here we will find the lowest cost node to expand next
    yet_to_visit_list = []
    # in this list we will put all node those already explored so that we don't explore it again
    visited_list = []

    # Add the start node
    yet_to_visit_list.append(start_node)

    # Adding a stop condition. This is to avoid any infinite loop and stop
    # execution after some reasonable number of steps
    outer_iterations = 0
    max_iterations = (len(maze) // 2) ** 10

    # what squares do we search . search movement is left-right-top-bottom
    # (4 movements) from every position

    move = [[-1, 0],  # go up
            [0, -1],  # go left
            [1, 0],  # go down
            [0, 1]]  # go right

    # find maze has got how many rows and columns
    no_rows, no_columns = np.shape(maze)

    # Loop until you find the end

    while len(yet_to_visit_list) > 0:

        # Every time any node is referred from yet_to_visit list, counter of limit operation incremented
        outer_iterations += 1

        # Get the current node
        current_node = yet_to_visit_list[0]
        current_index = 0
        for index, item in enumerate(yet_to_visit_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # if we hit this point return the path such as it may be no solution or
        # computation cost is too high
        if outer_iterations > max_iterations:
            print("giving up on pathfinding too many iterations")
            return return_path(current_node, maze)

        # Pop current node out off yet_to_visit list, add to visited list
        yet_to_visit_list.pop(current_index)
        visited_list.append(current_node)

        # test if goal is reached or not, if yes then return the path
        if current_node == end_node:
            return return_path(current_node, maze)

        # Generate children from all adjacent squares
        children = []

        for new_position in move:

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range (check if within maze boundary)
            if (node_position[0] > (no_rows - 1) or
                    node_position[0] < 0 or
                    node_position[1] > (no_columns - 1) or
                    node_position[1] < 0):
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)
            # print(new_node.position)
            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the visited list (search entire visited list)
            if len([visited_child for visited_child in visited_list if visited_child == child]) > 0:
                continue

            # Create the f, g, and h values
            child.g = current_node.g + cost
            # Heuristic costs calculated here, this is using eucledian distance
            child.h = (((child.position[0] - end_node.position[0]) ** 2) +
                       ((child.position[1] - end_node.position[1]) ** 2))

            child.f = child.g + child.h

            # Child is already in the yet_to_visit list and g cost is already lower
            if len([i for i in yet_to_visit_list if child == i and child.g > i.g]) > 0:
                continue

            # Add the child to the yet_to_visit list
            yet_to_visit_list.append(child)


def draw_circle(event, x, y, flags, param):
    global startx, starty, endx, endy, drawing, mode, chances

    if chances > 0:

        if chances == 1:
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                startx, starty = x, y
                chances = chances - 1
                print(x, y)

        else:
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                endx, endy = x, y
                chances = chances - 1
                print(x, y)


if __name__ == '__main__':

    img = cv2.imread("images.png", 0)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)
    dst = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
    xmax, ymax = dst.shape
    for i in range(0, xmax):
        for j in range(0, ymax):
            # listt.append(img[i, j])
            listt.append(dst[i, j])
    apple = np.reshape(listt, (xmax, ymax))
    # df1 = pd.DataFrame(apple).T
    # df1.to_excel(excel_writer="C:/Users/Guest/Desktop/Final2-1.xlsx")

    for i in range(xmax):
        for j in range(ymax):
            if 201 <= apple[i][j] <= 207:
                apple[i][j] = 1
            else:
                apple[i][j] = 0
    # apple=apple.transpose()
    # apple=np.flip()
    maze = apple.transpose()
    # df = pd.DataFrame(apple).T
    # df.to_excel(excel_writer="C:/Users/Guest/Desktop/Final2-22-temp.xlsx")

    #  Give your input here

    start = [32, 33]  # starting position  start = [32, 33]  # starting position-----end = [189, 165]  # ending position
    end = [189, 165]  # ending position

    cost = 1  # cost per movement
    path = search(apple, cost, start, end)
    # print(path)
    while 1:
        cv2.imshow('image', img)
        cv2.imshow("Gaussian Smoothing", np.hstack((img, dst)))
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == 27:
            break
cv2.destroyAllWindows()
