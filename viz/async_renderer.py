import multiprocessing
import numpy as np
from viz import renderer
from viz_utils.compare_dict import equal_dicts


class AsyncRenderer:
    def __init__(self):
        self._closed = False
        self._is_async = False
        self._cur_args = None
        self._cur_result = None
        self._cur_stamp = 0
        self._renderer_obj = None
        self._args_queue = None
        self._result_queue = None
        self._process = None
        self._cur_multiplier = None
        self._cur_background = None

    def close(self):
        self._closed = True
        self._renderer_obj = None
        if self._process is not None:
            self._process.terminate()
        self._process = None
        self._args_queue = None
        self._result_queue = None

    @property
    def is_async(self):
        return self._is_async

    def set_async(self, is_async):
        self._is_async = is_async

    @staticmethod
    def equal_dict(dict_1, dict_2):
        if dict_1 == None or dict_2 == None:
            return False
        for key, value in dict_1.items():
            if not np.array_equal(value, dict_2[key]):
                return False
        return True

    def set_args(self, **args):
        assert not self._closed
        something_changed = not equal_dicts(args, self._cur_args)
        if something_changed:
            if self._is_async:
                self._set_args_async(**args)
            else:
                self._set_args_sync(**args)
            self._cur_args = args

    def _set_args_async(self, **args):
        if self._process is None:
            self._args_queue = multiprocessing.Queue()
            self._result_queue = multiprocessing.Queue()
            try:
                multiprocessing.set_start_method("spawn")
            except RuntimeError:
                pass
            self._process = multiprocessing.Process(
                target=self._process_fn, args=(self._args_queue, self._result_queue), daemon=True
            )
            self._process.start()
        self._args_queue.put([args, self._cur_stamp])

    def _set_args_sync(self, **args):
        if self._renderer_obj is None:
            self._renderer_obj = renderer.Renderer()
        self._cur_result = self._renderer_obj.render(**args)

    def get_result(self):
        assert not self._closed
        if self._result_queue is not None:
            while self._result_queue.qsize() > 0:
                result, stamp = self._result_queue.get()
                if stamp == self._cur_stamp:
                    self._cur_result = result
        return self._cur_result

    def clear_result(self):
        assert not self._closed
        self._cur_args = None
        self._cur_result = None
        self._cur_stamp += 1

    @staticmethod
    def _process_fn(args_queue, result_queue):
        renderer_obj = renderer.Renderer()
        cur_args = None
        cur_stamp = None
        while True:
            args, stamp = args_queue.get()
            while args_queue.qsize() > 0:
                args, stamp = args_queue.get()
            if args != cur_args or stamp != cur_stamp:
                result = renderer_obj.render(**args)
                if "error" in result:
                    result.error = renderer.CapturedException(result.error)
                result_queue.put([result, stamp])
                cur_args = args
                cur_stamp = stamp
