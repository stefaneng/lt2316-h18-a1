class Node:
    def __init__(self):
        self.label = None
        self.attr = None
        self.children = {}

    def add_child(self, node, value):
        "Adds a child with the value of the label"
        self.children[value] = node

    def __str__(self):
        str = ""        
        return "Node(label = {}, attr = {}, children = {})".format(self.label, self.attr, self.children)
