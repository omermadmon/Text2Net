import unittest
from Text2Net import Text2Net


class Text2NetTest(unittest.TestCase):
    def test_text2net_contains_text_attribute(self):
        text = 'hello world'
        text2net = Text2Net(text)
        self.assertEqual(text2net.text, text)

    def test_text2net_assert_graph_nodes(self):
        text = 'hello world'
        text2net = Text2Net(text)
        G = text2net.transform(n_nodes=2)
        self.assertSetEqual(set(G.nodes), {'hello', 'world'})

    def test_text2net_weight_functions(self):
        text = 'maccabi haifa. hapoel haifa'
        text2net = Text2Net(text)
        G_indicator_weights = text2net.transform(n_nodes=3, weight_function='indicator')
        G_jaccard_weights = text2net.transform(n_nodes=3, weight_function='jaccard')
        self.assertEqual(G_indicator_weights['maccabi']['haifa']['weight'], 1)
        self.assertEqual(G_indicator_weights['hapoel']['haifa']['weight'], 1)
        self.assertEqual(G_jaccard_weights['maccabi']['haifa']['weight'], 0.5)
        self.assertEqual(G_jaccard_weights['hapoel']['haifa']['weight'], 0.5)
        self.assertRaises(ValueError, lambda: text2net.transform(n_nodes=3, weight_function='non_existing'))


if __name__ == '__main__':
    unittest.main()
