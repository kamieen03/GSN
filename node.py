from functions import FUN

class Node:
    def __init__(self, fun, children=[]):
        self.fun = fun
        self.children = []

    def __call__(self):
        inputs = arr([child.forward() for child in self.children])
        return self.fun(inputs)



def test():
    c = Node(lambda x: return 1)
    n = Node(FUN.SIN)
    print(n(1))


if __name__ == '__main__':
    test()
