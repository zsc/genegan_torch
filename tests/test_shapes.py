from __future__ import annotations

import unittest

import torch

from genegan.models.genegan import Discriminator, Joiner, Splitter


class TestShapes(unittest.TestCase):
    def test_splitter_shapes(self) -> None:
        b = 4
        x = torch.zeros(b, 3, 128, 128, dtype=torch.float32)
        splitter = Splitter(second_ratio=0.25)
        A, obj = splitter(x)
        self.assertEqual(tuple(A.shape), (b, 384, 8, 8))
        self.assertEqual(tuple(obj.shape), (b, 128, 8, 8))

    def test_joiner_shapes(self) -> None:
        b = 4
        A = torch.zeros(b, 384, 8, 8, dtype=torch.float32)
        obj = torch.zeros(b, 128, 8, 8, dtype=torch.float32)
        joiner = Joiner()
        out = joiner(A, obj)
        self.assertEqual(tuple(out.shape), (b, 3, 128, 128))

    def test_discriminator_shapes(self) -> None:
        b = 4
        x = torch.zeros(b, 3, 128, 128, dtype=torch.float32)
        d = Discriminator()
        out = d(x)
        self.assertEqual(tuple(out.shape), (b, 1))


if __name__ == "__main__":
    unittest.main()
