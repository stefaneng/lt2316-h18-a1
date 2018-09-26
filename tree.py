class Node:
    def __init__(self):
        self.label = None
        self.value = None
        self.child_attr = None
        self.attr = None
        self.children = {}
        self.depth = 0
        self.child_split = None
        self.continuous_child = False
        self.split_value = None

    def add_child(self, node, value):
        "Adds a child with the value of the label"
        self.children[value] = node

    def __str__(self):
        cont = "(Continuous)" if self.continuous_child else ""

        if not self.split_value:
            split_str = ""
        # True values are <= to the split_value
        elif self.value:
            split_str = "split = {} <= {}, ".format(self.attr, self.split_value)
        elif not self.value:
            split_str = "split = {} > {}, ".format(self.attr, self.split_value)

        string = "Node(label = {}, attr = {}, value = {}, {}child_attr = {}{}".format(self.label,  self.attr, self.value, split_str, cont, self.child_attr)
        for n in self.children.values():
            string += "," + "\n" + "  " * self.depth + str(n)
        string += ")"
        return string
