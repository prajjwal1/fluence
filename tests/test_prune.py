import unittest

from fluence.prune import Pruner

from fixtures.models import LeNet

class Test_Prune(unittest.TestCase):
    def test_pruner(self):
        model = LeNet()
        prune_proc = Pruner(model, ['conv', 'linear'], 0.675)
        prune_proc.perform_pruning('random')
        self.assertEqual(len(prune_proc.model.conv1._forward_pre_hooks), 1)
        prune_proc.make_permanent()
        self.assertEqual(len(prune_proc.model.conv1._forward_pre_hooks), 0)



