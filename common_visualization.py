from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
from networkx.drawing.nx_pydot import graphviz_layout


# positions for tree
def hierarchy_pos(G, root, levels=None, width=1., height=1.):
    '''If there is a cycle that is reachable from root, then this will see infinite recursion.
       G: the graph
       root: the root node
       levels: a dictionary
               key: level number (starting from 0)
               value: number of nodes in this level
       width: horizontal space allocated for drawing
       height: vertical space allocated for drawing'''
    TOTAL = "total"
    CURRENT = "current"

    def make_levels(levels, node=root, currentLevel=0, parent=None):
        """Compute the number of nodes for each level
        """
        if not currentLevel in levels:
            levels[currentLevel] = {TOTAL: 0, CURRENT: 0}
        levels[currentLevel][TOTAL] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                levels = make_levels(levels, neighbor, currentLevel + 1, node)
        return levels

    def make_pos(pos, node=root, currentLevel=0, parent=None, vert_loc=0):
        dx = 1 / levels[currentLevel][TOTAL]
        left = dx / 2
        pos[node] = ((left + dx * levels[currentLevel][CURRENT]) * width, vert_loc)
        levels[currentLevel][CURRENT] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                pos = make_pos(pos, neighbor, currentLevel + 1, node, vert_loc - vert_gap)
        return pos

    if levels is None:
        levels = make_levels({})
    else:
        levels = {l: {TOTAL: levels[l], CURRENT: 0} for l in levels}
    vert_gap = height / (max([l for l in levels]) + 1)
    return make_pos({})


# prepare binary with specific target probability > 0.5 or maximal among all (any source, specific target) pair
def prepare_binary(m0, m1, t_binary, plot=False):
    # m0_binary = m0 > t_binary
    # m1_binary = m1 > t_binary
    #
    # for target in range(m0.shape[1], m0.shape[0]):
    #     if m0_binary[target, :].sum() + m1_binary[target, :].sum() < 1:
    #         if m0[target, :].max() > m1[target, :].max():
    #             m0_binary[target, m0[target, :].argmax()] = True
    #         else:
    #             m1_binary[target, m1[target, :].argmax()] = True

    m0_binary = np.zeros_like(m0).astype(bool)
    m1_binary = np.zeros_like(m0).astype(bool)

    for source in range(m0.shape[1]):
        m0_binary[m0[:, source].argmax(), source] = True
        m1_binary[m1[:, source].argmax(), source] = True

    if plot:
        fig = plt.figure(figsize=(8, 8))
        fig.add_subplot(121)
        img = plt.imshow(m0_binary, interpolation='nearest')
        img.set_cmap(cmap)
        plt.axis('off')
        fig.add_subplot(122)
        img = plt.imshow(m1_binary, interpolation='nearest')
        img.set_cmap(cmap)
        plt.axis('off')
        plt.show()
        plt.close()

    return m0_binary, m1_binary


def draw_graph_as_tree(m0, m1, labels, t_binary=0.5, pos_type='top-down', root=0):
    m0_binary, m1_binary = prepare_binary(m0, m1, t_binary, plot=False)

    g = nx.DiGraph(directed=True)

    # define nodes and edges existing in binary version of m0 and m1
    node_list = range(m0.shape[1])
    leaf_list = range(m0.shape[1], m0.shape[0])
    edge_list_0 = []
    edge_list_1 = []
    edge_color_0 = []
    edge_color_1 = []
    for source in range(m0.shape[1]):
        for target in range(m0.shape[0]):
            if m0_binary[target, source]:
                edge_list_0.append((source, target))
                edge_color_0.append(m0[target, source])
            if m1_binary[target, source]:
                edge_list_1.append((source, target))
                edge_color_1.append(m1[target, source])

    # add nodes and edges to graph
    g.add_nodes_from(node_list)
    g.add_nodes_from(leaf_list)
    g.add_edges_from(edge_list_0)
    g.add_edges_from(edge_list_1)

    # generate DFS tree from graph (this operation removes nodes and edges unreachable from the root)
    g = nx.bfs_tree(g, root)

    if pos_type == 'top-down':
        # obtain hierachy positions of the tree
        pos = hierarchy_pos(g, root)
    elif pos_type == 'circle':
        pos = graphviz_layout(g, prog="twopi", root=root)

    # update nodes and edges so that they correspond only to nodes reachable from the root
    node_list = [node_list[i] for i in range(len(node_list)) if node_list[i] in list(g.nodes())]
    leaf_list = [leaf_list[i] for i in range(len(leaf_list)) if leaf_list[i] in list(g.nodes())]
    edge_list_0 = []
    edge_list_0_back = []  # backed edges
    edge_list_1 = []
    edge_list_1_back = []
    edge_color_0 = []
    edge_color_0_back = []
    edge_color_1 = []
    edge_color_1_back = []
    for source in range(m0.shape[1]):
        for target in range(1, m0.shape[0]):
            if m0_binary[target, source] and source in list(g.nodes()) and target in list(g.nodes()):
                if (source, target) in list(g.edges()):
                    edge_list_0.append((source, target))
                    edge_color_0.append(m0[target, source])
                else:
                    edge_list_0_back.append((source, target))
                    edge_color_0_back.append(m0[target, source])
            if m1_binary[target, source] and source in list(g.nodes()) and target in list(g.nodes()):
                if (source, target) in list(g.edges()):
                    edge_list_1.append((source, target))
                    edge_color_1.append(m1[target, source])
                else:
                    edge_list_1_back.append((source, target))
                    edge_color_1_back.append(m1[target, source])

    # name leafs like the MNIST digits
    leaf_list_names = {}
    for i in leaf_list:
        leaf_list_names[i] = labels[i - m0.shape[1]]
    node_list_names = {}
    for i in node_list:
        node_list_names[i] = i

    # draw graph with different color of nodes and different colors of edges (depending on their origin)
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(g, pos, nodelist=node_list, node_color="tab:brown")
    nx.draw_networkx_nodes(g, pos, nodelist=leaf_list, node_color="tab:green")
    nx.draw_networkx_labels(g, pos, labels=leaf_list_names)
    nx.draw_networkx_labels(g, pos, labels=node_list_names)
    nx.draw_networkx_edges(g, pos, edgelist=edge_list_0, width=3,
                           edge_color=edge_color_0, edge_cmap=plt.cm.Reds, edge_vmin=0,
                           edge_vmax=1)  # connectionstyle='arc3, rad = -0.1')
    nx.draw_networkx_edges(g, pos, edgelist=edge_list_0_back, width=1, style='dotted',
                           edge_color=edge_color_0_back, edge_cmap=plt.cm.Reds, edge_vmin=0, edge_vmax=1)
    nx.draw_networkx_edges(g, pos, edgelist=edge_list_1, width=3,
                           edge_color=edge_color_1, edge_cmap=plt.cm.Greens, edge_vmin=0, edge_vmax=1)
    nx.draw_networkx_edges(g, pos, edgelist=edge_list_1_back, width=1, style='dotted',
                           edge_color=edge_color_1_back, edge_cmap=plt.cm.Greens, edge_vmin=0, edge_vmax=1)

    return g, pos
