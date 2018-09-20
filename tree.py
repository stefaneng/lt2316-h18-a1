class Node:
    def __init__(self):
        self.label = None
        self.value = None
        self.child_attr = None
        self.attr = None
        self.children = {}
        self.depth = 0

    def add_child(self, node, value):
        "Adds a child with the value of the label"
        self.children[value] = node

    def __str__(self):
        string = "Node(label = {}, attr = {}, value = {}, child_attr = {}".format(self.label, self.attr, self.value, self.child_attr)
        for n in self.children.values():
            string += "," + "\n" + "  " * self.depth + str(n)
        string += ")"
        return string
