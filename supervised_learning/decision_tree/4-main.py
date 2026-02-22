$ cat 4-main.py
#!/usr/bin/env python3

Node = __import__('4-build_decision_tree').Node
Leaf = __import__('4-build_decision_tree').Leaf
Decision_Tree = __import__('4-build_decision_tree').Decision_Tree

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

def test_bounds():
    c = 0
    for T in [example_0(), example_1(5)]:
        print("example_", c)
        c += 1
        T.update_bounds()
        leaves = T.get_leaves()
        for i in range(len(leaves)):
            print("  leaf number ", i)
            print("    lower :", leaves[i].lower)
            print("    upper :", leaves[i].upper)

test_bounds()
$ ./4-main.py
example_ 0
  leaf number  0
    lower : {0: 0.5}
    upper : {0: inf}
  leaf number  1
    lower : {0: -inf, 1: 30000}
    upper : {0: 0.5}
  leaf number  2
    lower : {0: -inf}
    upper : {0: 0.5, 1: 30000}
example_ 1
  leaf number  0
    lower : {0: 30.5}
    upper : {0: inf}
  leaf number  1
    lower : {0: 29.5}
    upper : {0: 30.5}
  leaf number  2
    lower : {0: 28.5}
    upper : {0: 29.5}
  leaf number  3
    lower : {0: 27.5}
    upper : {0: 28.5}
  leaf number  4
    lower : {0: 26.5}
    upper : {0: 27.5}
  leaf number  5
    lower : {0: 25.5}
    upper : {0: 26.5}
  leaf number  6
    lower : {0: 24.5}
    upper : {0: 25.5}
  leaf number  7
    lower : {0: 23.5}
    upper : {0: 24.5}
  leaf number  8
    lower : {0: 22.5}
    upper : {0: 23.5}
  leaf number  9
    lower : {0: 21.5}
    upper : {0: 22.5}
  leaf number  10
    lower : {0: 20.5}
    upper : {0: 21.5}
  leaf number  11
    lower : {0: 19.5}
    upper : {0: 20.5}
  leaf number  12
    lower : {0: 18.5}
    upper : {0: 19.5}
  leaf number  13
    lower : {0: 17.5}
    upper : {0: 18.5}
  leaf number  14
    lower : {0: 16.5}
    upper : {0: 17.5}
  leaf number  15
    lower : {0: 15.5}
    upper : {0: 16.5}
  leaf number  16
    lower : {0: 14.5}
    upper : {0: 15.5}
  leaf number  17
    lower : {0: 13.5}
    upper : {0: 14.5}
  leaf number  18
    lower : {0: 12.5}
    upper : {0: 13.5}
  leaf number  19
    lower : {0: 11.5}
    upper : {0: 12.5}
  leaf number  20
    lower : {0: 10.5}
    upper : {0: 11.5}
  leaf number  21
    lower : {0: 9.5}
    upper : {0: 10.5}
  leaf number  22
    lower : {0: 8.5}
    upper : {0: 9.5}
  leaf number  23
    lower : {0: 7.5}
    upper : {0: 8.5}
  leaf number  24
    lower : {0: 6.5}
    upper : {0: 7.5}
  leaf number  25
    lower : {0: 5.5}
    upper : {0: 6.5}
  leaf number  26
    lower : {0: 4.5}
    upper : {0: 5.5}
  leaf number  27
    lower : {0: 3.5}
    upper : {0: 4.5}
  leaf number  28
    lower : {0: 2.5}
    upper : {0: 3.5}
  leaf number  29
    lower : {0: 1.5}
    upper : {0: 2.5}
  leaf number  30
    lower : {0: 0.5}
    upper : {0: 1.5}
  leaf number  31
    lower : {0: -inf}
    upper : {0: 0.5}