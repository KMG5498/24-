# @title 트리 클래스 정의
import random

class Treenode:
    def __init__(self, value, parent, depth):
        self.value = value
        self.left = None
        self.right = None
        self.parent = parent
        self.depth = depth


def random_making_tree(terminal_node_list, function_node_list, min_depth, max_depth, now_depth, parent_node):
    def r_f():
        return random.choice(function_node_list)
    def r_tf():
        return random.choice(terminal_node_list + function_node_list)
    def r_t():
        return random.choice(terminal_node_list)
    t = terminal_node_list
    f = function_node_list

    if now_depth == 1:
        node = Treenode(r_f(), None, 1)
    elif now_depth >= max_depth:
        node = Treenode(r_t(), parent_node, now_depth)
        return node
    else:
        node = Treenode(r_tf(), parent_node, now_depth)

    if node.value in ['+', '-', '*']:
        left_node = random_making_tree(t, f, min_depth, max_depth, now_depth+1, node)
        right_node = random_making_tree(t, f, min_depth, max_depth, now_depth+1, node)
        node.left = left_node
        node.right = right_node
    elif node.value == 'neg' or node.value == 'is_positive':
        child_node = random_making_tree(t, f, min_depth, max_depth, now_depth+1, node)
        node.left = child_node
    else:
        return node
    return node

def translate_to_priority(node, value_dict):
    if node.value in ['+', '-', '*']:
        if node.value == '+':
            return translate_to_priority(node.left, value_dict) + translate_to_priority(node.right, value_dict)
        elif node.value == '-':
            return translate_to_priority(node.left, value_dict) - translate_to_priority(node.right, value_dict)
        else:
            return translate_to_priority(node.left, value_dict) * translate_to_priority(node.right, value_dict)
    elif node.value == 'neg':
        return -translate_to_priority(node.left, value_dict)
    elif node.value == 'is_positive':
        return max(translate_to_priority(node.left, value_dict), 0)
    elif node.value ==None:
        pass
    else:
        return value_dict[node.value]

def copy_tree(root, parent):
    if root != None:
        new_root = Treenode(root.value, parent, root.depth)
        new_root.left = copy_tree(root.left, new_root)
        new_root.right = copy_tree(root.right, new_root)
        return new_root
    else:
        pass

def print_tree(node):
    if node != None:
        print(node.value)
        print_tree(node.left)
        print_tree(node.right)
    else:
        pass

# 트리를 리스트로 만들기 => 트리 처리를 더 용이하게 하려고
def making_list_from_node(node, node_list):
    if node.value != None:
        node_list.append(node)
        if node.left != None:
            node_list = node_list + making_list_from_node(node.left, [])
        if node.right != None:
            node_list = node_list + making_list_from_node(node.right, [])
    return node_list

# 트리에서 특정 노드의 서브트리를 구하는 기능
def get_subtree(node):
    return node, node.parent

# 트리에서 특정 노드가 부모노드의 어떤 방향인지 구하는 기능
def remove_subtree(node):
    parent = node.parent
    if parent.left.value == node.value:
        return 'left'
    else:
        return 'right'


def crossover(node1, node2):
    root1 = copy_tree(node1, None)
    root2 = copy_tree(node2, None)
    tree1 = making_list_from_node(root1, [])
    tree2 = making_list_from_node(root2, [])
    tree1.remove(root1)
    tree2.remove(root2)
    point1 = tree1[random.randint(0, len(tree1)-1)]
    point2 = tree2[random.randint(0, len(tree2)-1)]
    parent1 = point1.parent
    parent2 = point2.parent
    direction1 = remove_subtree(point1)
    direction2 = remove_subtree(point2)
    point2.parent = parent1
    point1.parent = parent2
    if direction1 == 'left':
        parent1.left = point2
    else:
        parent1.right = point2
    if direction2 == 'left':
        parent2.left = point1
    else:
        parent2.right = point1
    return root1, root2

def mutation(node,terminal_node_list, function_node_list, mutation_minimum, mutation_maximum):
    root = copy_tree(node, None)
    tree = making_list_from_node(root, [])
    tree.remove(root)
    point = tree[random.randint(0, len(tree)-1)]
    parent = point.parent
    new_node = random_making_tree(terminal_node_list, function_node_list, mutation_minimum, mutation_maximum, 1, None)
    direction = remove_subtree(point)
    if direction == 'left':
        parent.left = new_node
    else:
        parent.right = new_node
    new_node.parent = parent
    return root

def making_random_population(terminal_node_list, function_node_list, min_depth, max_depth, num_population):
    population = []
    for _ in range(num_population):
        population.append(random_making_tree(terminal_node_list, function_node_list, min_depth, max_depth, 1, None))
    return population
