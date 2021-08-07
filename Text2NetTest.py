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


if __name__ == '__main__':
    unittest.main()
