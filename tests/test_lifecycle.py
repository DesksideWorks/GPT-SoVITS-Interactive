"""Fast lifecycle regression tests; no GPU, checkpoint or server needed."""
import asyncio
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import api_server
from run_inference import GPTSoVITSInference
from GPT_SoVITS.AR.modules.embedding import SinePositionalEmbedding


class LifecycleTests(unittest.TestCase):
    def test_runtime_import_does_not_load_training_framework(self):
        self.assertNotIn("pytorch_lightning", sys.modules)
        self.assertNotIn("torchmetrics", sys.modules)

    def engine(self):
        engine = GPTSoVITSInference.__new__(GPTSoVITSInference)
        engine.is_hibernated = False
        engine.lifecycle_error = None
        engine.backend = mock.Mock()
        engine.gpt_backend = engine.backend
        engine._move_runtime = mock.Mock()
        return engine

    def test_sleep_and_wake_are_idempotent(self):
        engine = self.engine()
        engine.wake()
        engine._move_runtime.assert_not_called()
        engine.hibernate()
        engine.hibernate()
        self.assertTrue(engine.is_hibernated)
        engine._move_runtime.assert_called_once_with(sleeping=True)
        engine.backend.empty_cache.assert_called_once()
        engine.wake()
        engine.wake()
        self.assertFalse(engine.is_hibernated)
        self.assertEqual(2, engine._move_runtime.call_count)

    def test_partial_sleep_failure_is_never_reported_as_ready(self):
        engine = self.engine()
        engine._move_runtime.side_effect = RuntimeError("copy failed")
        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            engine.hibernate()
        self.assertIn("hibernation failed", engine.lifecycle_error)
        with self.assertRaisesRegex(RuntimeError, "hibernation failed"):
            engine.wake()
        self.assertEqual(1, engine._move_runtime.call_count)

    def test_partial_wake_failure_remains_failed(self):
        engine = self.engine()
        engine.is_hibernated = True
        engine._move_runtime.side_effect = RuntimeError("out of memory")
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            engine.wake()
        self.assertTrue(engine.is_hibernated)
        self.assertIn("wake failed", engine.lifecycle_error)

    def test_position_cache_moves_but_does_not_change_checkpoint_keys(self):
        position = SinePositionalEmbedding(16)
        self.assertNotIn("pe", position.state_dict())
        self.assertIn("pe", dict(position.named_buffers()))
        position.to(dtype=torch.float64)
        self.assertEqual(torch.float64, position.pe.dtype)

    def test_status_does_not_load_or_wake_model(self):
        manager = mock.Mock()
        manager.active_engine = types.SimpleNamespace(is_hibernated=True, lifecycle_error=None,
                                                      sample_rate=32000, startup_metrics={})
        with mock.patch.object(api_server, "model_manager", manager):
            status = api_server.runtime_status()
        self.assertEqual("hibernated", status["runtime_state"])
        self.assertFalse(status["ready"])
        manager.get_engine.assert_not_called()
        self.assertEqual([], manager.method_calls)

    def test_failed_engine_status_is_error(self):
        manager = mock.Mock()
        manager.active_engine = types.SimpleNamespace(is_hibernated=False, lifecycle_error="failed move",
                                                      sample_rate=32000, startup_metrics={})
        with mock.patch.object(api_server, "model_manager", manager):
            status = api_server.runtime_status()
        self.assertEqual("error", status["runtime_state"])
        self.assertFalse(status["ready"])

    def test_cached_engine_wakes_without_constructing_another(self):
        manager = api_server.ModelManager("cpu")
        fake = mock.Mock()
        with mock.patch("run_inference.GPTSoVITSInference", return_value=fake) as constructor:
            first = manager.get_engine("gpt", "sovits", "prompt")
            second = manager.get_engine("gpt", "sovits", "prompt")
        self.assertIs(first, second)
        constructor.assert_called_once()
        self.assertEqual(2, fake.wake.call_count)


class SerializationTests(unittest.IsolatedAsyncioTestCase):
    async def test_lock_covers_streaming_body_but_status_is_read_only(self):
        entered = asyncio.Event()
        release_stream = asyncio.Event()
        calls = []

        async def app(scope, receive, send):
            calls.append(scope["path"])
            if scope["path"] == "/stream":
                await send({"type": "http.response.start", "status": 200, "headers": []})
                entered.set()
                await release_stream.wait()
                await send({"type": "http.response.body", "body": b"last", "more_body": False})

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            pass

        middleware = api_server.SerializedTtsOperations(app)
        stream = asyncio.create_task(middleware({"type": "http", "method": "POST", "path": "/stream"}, receive, send))
        await entered.wait()
        sleep = asyncio.create_task(middleware({"type": "http", "method": "POST", "path": "/hibernate"}, receive, send))
        await middleware({"type": "http", "method": "GET", "path": "/status"}, receive, send)
        await asyncio.sleep(0)
        self.assertEqual(["/stream", "/status"], calls)
        release_stream.set()
        await asyncio.gather(stream, sleep)
        self.assertEqual(["/stream", "/status", "/hibernate"], calls)


if __name__ == "__main__":
    unittest.main()
