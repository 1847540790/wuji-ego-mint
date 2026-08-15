from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from mint.inference.acceleration import (
    COMPILE_MODES,
    _eligible_fp8_linear,
    compile_hotspots,
    normalize_optional_mode,
    prepare_allocator_for_compile,
    resolve_compile_mode,
    resolve_fp8_mode,
)
from mint.inference.engine import StudentEngine


class InferenceAccelerationTest(unittest.TestCase):
    def test_optional_modes_accept_auto_and_off(self):
        self.assertIsNone(
            normalize_optional_mode("off", choices=COMPILE_MODES, name="compile")
        )
        self.assertEqual(
            normalize_optional_mode("AUTO", choices=COMPILE_MODES, name="compile"),
            "auto",
        )
        with self.assertRaisesRegex(ValueError, "Unsupported compile"):
            normalize_optional_mode("fastest", choices=COMPILE_MODES, name="compile")

    def test_compile_auto_requires_cuda(self):
        cpu = SimpleNamespace(type="cpu")
        cuda = SimpleNamespace(type="cuda")
        self.assertIsNone(resolve_compile_mode("auto", [cpu])[0])
        self.assertEqual(resolve_compile_mode("auto", [cuda])[0], "reduce-overhead")

    def test_fp8_auto_requires_compile(self):
        cuda = SimpleNamespace(type="cuda")
        mode, reason = resolve_fp8_mode("auto", [cuda], None)
        self.assertIsNone(mode)
        self.assertIn("requires torch.compile", reason)

    def test_cudagraph_compile_removes_expandable_segments_before_cuda(self):
        original = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            "garbage_collection_threshold:0.8,expandable_segments:True"
        )
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False))
        try:
            with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                self.assertTrue(prepare_allocator_for_compile("reduce-overhead"))
            self.assertNotIn("expandable_segments", os.environ["PYTORCH_CUDA_ALLOC_CONF"])
        finally:
            if original is None:
                os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            else:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = original

    def test_fp8_filter_only_selects_large_aligned_aggregator_linears(self):
        eligible = torch.nn.Linear(1024, 4096)
        self.assertTrue(
            _eligible_fp8_linear(eligible, "backbone.aggregator.blocks.0.mlp.fc1")
        )
        self.assertFalse(_eligible_fp8_linear(torch.nn.Linear(128, 128), "backbone.aggregator.small"))
        self.assertFalse(_eligible_fp8_linear(torch.nn.Linear(1024, 218), "backbone.aggregator.output"))
        self.assertFalse(_eligible_fp8_linear(eligible, "hand_head.layers.0"))

    def test_compile_window_batch_keeps_local_batch_one(self):
        engine = StudentEngine.__new__(StudentEngine)
        engine.model = object()
        engine.models = [object()] * 4
        engine.compile_mode = "reduce-overhead"

        self.assertEqual(engine._effective_window_batch_size(4), 4)
        self.assertEqual(engine._effective_window_batch_size(8), 4)

        engine.models = [object()]
        self.assertEqual(engine._effective_window_batch_size(4), 1)

    def test_compiled_hotspot_uses_eager_module_for_a_new_shape(self):
        class CountingBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, value):
                self.calls += 1
                return value + 1

        class CompiledProxy(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                self.calls = 0

            def forward(self, *args, **kwargs):
                self.calls += 1
                return self.module(*args, **kwargs)

        block = CountingBlock()
        aggregator = SimpleNamespace(
            frame_blocks=torch.nn.ModuleList([block]),
            patch_embed=SimpleNamespace(blocks=None),
            global_blocks=[],
            use_sdpa=True,
        )
        model = SimpleNamespace(
            backbone=SimpleNamespace(aggregator=aggregator)
        )
        proxies = []

        def fake_compile(module, **_kwargs):
            proxy = CompiledProxy(module)
            proxies.append(proxy)
            return proxy

        with mock.patch("torch.compile", side_effect=fake_compile):
            compile_hotspots(model, "reduce-overhead")
        wrapped = aggregator.frame_blocks[0]
        wrapped(torch.zeros(2, 3))
        wrapped(torch.zeros(2, 3))
        wrapped(torch.zeros(1, 3))

        self.assertEqual(proxies[0].calls, 2)
        self.assertEqual(block.calls, 3)


if __name__ == "__main__":
    unittest.main()
