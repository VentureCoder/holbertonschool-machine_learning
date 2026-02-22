$ cat 2-main.py
#!/usr/bin/env python3

Node = __import__('2-build_decision_tree').Node
Leaf = __import__('2-build_decision_tree').Leaf
Decision_Tree = __import__('2-build_decision_tree').Decision_Tree

def example_0():
    leaf0 = Leaf(0, depth=1)
    leaf1 = Leaf(0, depth=2)
    leaf2 = Leaf(1, depth=2)
    internal_node = Node(feature=1, threshold=30000, left_child=leaf1, right_child=leaf2, depth=1)
    root = Node(feature=0, threshold=.5, left_child=leaf0, right_child=internal_node, depth=0, is_root=True)
    return Decision_Tree(root=root)


def example_1(depth):
    level = [Leaf(i, depth=depth) for i in range(2 ** depth)]
    level.reverse()

    def get_v(node):
        if node.is_leaf:
            return node.value
        else:
            return node.threshold

    for d in range(depth):
        level = [Node(feature=0,
                      threshold=(get_v(level[2 * i]) + get_v(level[2 * i + 1])) / 2,
                      left_child=level[2 * i],
                      right_child=level[2 * i + 1], depth=depth - d - 1) for i in range(2 ** (depth - d - 1))]
    root = level[0]
    root.is_root = True
    return Decision_Tree(root=root)

#Print Tree example 0
print(example_0())
#Print Tree example 1
print(example_1(4))
$ ./2-main.py
root [feature=0, threshold=0.5]
    +---> leaf [value=0]
    +---> node [feature=1, threshold=30000]
           +---> leaf [value=0]
           +---> leaf [value=1]

root [feature=0, threshold=7.5]
    +---> node [feature=0, threshold=11.5]
    |      +---> node [feature=0, threshold=13.5]
    |      |      +---> node [feature=0, threshold=14.5]
    |      |      |      +---> leaf [value=15]
    |      |      |      +---> leaf [value=14]
    |      |      +---> node [feature=0, threshold=12.5]
    |      |             +---> leaf [value=13]
    |      |             +---> leaf [value=12]
    |      +---> node [feature=0, threshold=9.5]
    |             +---> node [feature=0, threshold=10.5]
    |             |      +---> leaf [value=11]
    |             |      +---> leaf [value=10]
    |             +---> node [feature=0, threshold=8.5]
    |                    +---> leaf [value=9]
    |                    +---> leaf [value=8]
    +---> node [feature=0, threshold=3.5]
           +---> node [feature=0, threshold=5.5]
           |      +---> node [feature=0, threshold=6.5]
           |      |      +---> leaf [value=7]
           |      |      +---> leaf [value=6]
           |      +---> node [feature=0, threshold=4.5]
           |             +---> leaf [value=5]
           |             +---> leaf [value=4]
           +---> node [feature=0, threshold=1.5]
                  +---> node [feature=0, threshold=2.5]
                  |      +---> leaf [value=3]
                  |      +---> leaf [value=2]
                  +---> node [feature=0, threshold=0.5]
                         +---> leaf [value=1]
                         +---> leaf [value=0]