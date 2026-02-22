from __future__ import annotations

import unittest

import torch

from genegan.models.genegan import Discriminator, Joiner, Splitter


class TestShapes(unittest.TestCase):
    def test_splitter_shapes(self) -> None:
        b = 4
        splitter = Splitter(second_ratio=0.25)
        for img_size in (64, 96, 128):
            x = torch.zeros(b, 3, img_size, img_size, dtype=torch.float32)
            A, obj = splitter(x)
            s = 12 if img_size == 96 else 16
            self.assertEqual(tuple(A.shape), (b, 384, s, s))
            self.assertEqual(tuple(obj.shape), (b, 128, s, s))

    def test_splitter_shapes_obj_blockconv(self) -> None:
        b = 4
        splitter = Splitter(second_ratio=0.25, obj_blockconv=True, obj_block_size=4)
        for img_size in (64, 96, 128):
            x = torch.zeros(b, 3, img_size, img_size, dtype=torch.float32)
            A, obj = splitter(x)
            s = 12 if img_size == 96 else 16
            self.assertEqual(tuple(A.shape), (b, 384, s, s))
            self.assertEqual(tuple(obj.shape), (b, 128, s, s))

    def test_joiner_shapes(self) -> None:
        b = 4
        joiner = Joiner()
        A16 = torch.zeros(b, 384, 16, 16, dtype=torch.float32)
        obj16 = torch.zeros(b, 128, 16, 16, dtype=torch.float32)
        out64 = joiner(A16, obj16, out_size=64)
        self.assertEqual(tuple(out64.shape), (b, 3, 64, 64))
        out128 = joiner(A16, obj16, out_size=128)
        self.assertEqual(tuple(out128.shape), (b, 3, 128, 128))

        A12 = torch.zeros(b, 384, 12, 12, dtype=torch.float32)
        obj12 = torch.zeros(b, 128, 12, 12, dtype=torch.float32)
        out96 = joiner(A12, obj12, out_size=96)
        self.assertEqual(tuple(out96.shape), (b, 3, 96, 96))

    def test_discriminator_shapes(self) -> None:
        b = 4
        d = Discriminator()
        for img_size in (64, 96, 128):
            x = torch.zeros(b, 3, img_size, img_size, dtype=torch.float32)
            out = d(x)
            self.assertEqual(tuple(out.shape), (b, 1))


if __name__ == "__main__":
    unittest.main()
