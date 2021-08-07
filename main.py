from Text2Net import Text2Net
from Utils import visualize

if __name__ == '__main__':
    text = 'These pretzels are making me thirsty!'
    text2net = Text2Net(text)
    G = text2net.transform(n_nodes=6)
    visualize(G, edges_factor=5, nodes_factor=1000)
