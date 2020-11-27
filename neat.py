from functions import FUN
from node import node, net


class Neat:
    def __init__(self, pop_size, n_inputs, n_outputs):
        self.population = [Net(W, n_inputs, n_outputs) for _ in range(pop_size)]
    
    def change_activation(self, net):
        node = random.choice(net.nodes)
        node.fun = random.choice(list(FUN))

    def insert_node(self, net):
        parent = random.choice(net.nodes + net.ouputs)
        child = random.choice(parent.children)
        node = Node(random.choice(list(FUN)))

        node.children.append(child)
        parent.children.remove(child)
        parent.children.append(node)

        net.nodes.append(node)
        #TODO: add new node to the proper net layer

    def add_connection(self, net):
        l1, l2 = sorted(random.sample(range(len(net.layers))+2, k=2))
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
