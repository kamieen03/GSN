from functions import FUN
from node import node, net


class Neat:
    def __init__(self, pop_size, n_inputs, n_outputs):
        self.population = [Net(n_inputs, n_outputs) for _ in range(pop_size)]
    
    def change_activation(self, net):
        node = random.choice(net.nodes)
        node.fun = random.choice(list(FUN))

    def insert_node(self, net):
        # get random node in  network and number of it's layer
        weights = [len(layer) for layer in net.layers] + [len(net.ouputs)]
        lparent = random.choices(range(len(net.layers))+1, weights = weights)
        if lparent < len(net.layers):
            parent = random.choice(net.layers[lparent])
        else:
            parent = random.choice(net.ouputs)

        # get one of it's children and produce new node between parent and child
        child = random.choice(parent.children)
        node = Node(random.choice(list(FUN)))
        node.children.append(child)
        parent.children.remove(child)
        parent.children.append(node)

        # insert newly created node to list of nodes and add it to proper layer
        # if parent and child were in consecutive layer, then we have to create a new layerr between them
        net.nodes.append(node)
        if child in net.layers[lparent-1]:
            net.layers.insert(lparent, [])
            net.layers[lparent].append(node)
        else:
            net.layers[lparent-1].append(node)

    def add_connection(self, net):
        # draw random layers according to numbers of nodes in them
        weights1 = [len(net.inputs)] + [len(layer) for layer in net.layers] + [len(net.ouputs)]
        l1 = random.choices(range(len(net.layers))+2, weights=weights1)
        possible_layers = list(range(len(net.layers))+2)
        possible_layers.pop(l1)
        weights2 = weights1
        weights2.pop(l1)
        l2 = random.choices(possible_layers, weights=weights2)
        
        # draw random nodes from chosen layers; node n1 must be in layer earlier than node n2
        l1, l2 = sorted([l1,l2])
        if l1 == 0:
            n1 = random.choice(net.inputs)
        else:
            n1 = random.choice(net.layers[l1-1])
        if l2 == len(net.layers) + 1:
            n2 = random.choice(net.outputs)
        else:
            n2 = random.choice(net.layers[l2-1])
        if n1 not in n2.children:
            n2.children.append(n1)



    def run(self):
       pass 
