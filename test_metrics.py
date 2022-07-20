import torch
import unittest

from train_mtl import single_label_metrics


class TestMetrics(unittest.TestCase):
    def test_perfect_labels(self):
        self.input_predictions = torch.Tensor([[1,0,0],[1,0,0,],[1,0,0,],[1,0,0,],[1,0,0,]])
        self.gold_labels = torch.Tensor([0,0,0,0,0])
        metrics = single_label_metrics(self.input_predictions, self.gold_labels)
        self.assertEqual(metrics['macro f1'], 1.0)
        self.assertEqual(metrics['accuracy'], 1.0)

    def test_wrong_labels(self):
        self.input_predictions = torch.Tensor([[1,0,0],[1,0,0,],[1,0,0,],[1,0,0,],[1,0,0,]])
        self.gold_labels = torch.Tensor([1,1,1,1,1])
        metrics = single_label_metrics(self.input_predictions, self.gold_labels)
        self.assertEqual(metrics['macro f1'], 0.0)
        self.assertEqual(metrics['accuracy'], 0.0)

    def test_half_labels(self):
        self.input_predictions = torch.Tensor([[1,0,0],[1,0,0,],[1,0,0,],[1,0,0,],[1,0,0,]])
        self.gold_labels = torch.Tensor([1,1,0,0,0])
        metrics = single_label_metrics(self.input_predictions, self.gold_labels)
        self.assertAlmostEqual(metrics['macro f1'], 0.375, places=3)
        self.assertEqual(metrics['accuracy'], 0.6)
