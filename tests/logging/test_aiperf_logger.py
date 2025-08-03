# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import time
import timeit
from collections.abc import Callable
from unittest.mock import Mock

import pytest

from aiperf.common.aiperf_logger import (
    _CRITICAL,
    _DEBUG,
    _ERROR,
    _INFO,
    _NOTICE,
    _SUCCESS,
    _TRACE,
    _WARNING,
    AIPerfLogger,
)
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import CreditPhase
from aiperf.common.models import RequestRecord, TextResponse
from tests.utils.time_traveler import TimeTraveler


@pytest.fixture
def aiperf_logger():
    aiperf_logger = AIPerfLogger("test_aiperf_logger")
    aiperf_logger.set_level(_INFO)
    for handler in aiperf_logger.root.handlers[:]:
        aiperf_logger.root.removeHandler(handler)
    aiperf_logger.addHandler(logging.NullHandler())
    yield aiperf_logger


@pytest.fixture
def standard_logger():
    logger = logging.getLogger("test_standard_logger")
    logger.setLevel(_INFO)
    for handler in logger.root.handlers[:]:
        logger.root.removeHandler(handler)
    logger.addHandler(logging.NullHandler())
    yield logger


@pytest.fixture
def large_message():
    return RequestRecord(
        request={
            "id": "123",
            "url": "http://localhost:8080",
            "method": "GET",
            "headers": {
                "Content-Type": "application/json",
            },
        },
        timestamp_ns=time.time_ns(),
        start_perf_ns=time.perf_counter_ns(),
        end_perf_ns=time.perf_counter_ns() + (1_000_000_000 * 101),
        recv_start_perf_ns=time.perf_counter_ns() + 1_000_000_000,
        status=200,
        responses=[
            TextResponse(
                perf_ns=time.perf_counter_ns() + (i * 1_000_000_000),
                text="Hello, world!",
            )
            for i in range(1, 101)
        ],
        error=None,
        delayed_ns=None,
        credit_phase=CreditPhase.PROFILING,
    )


def compare_logger_performance(
    aiperf_logger_func,
    standard_logger_func,
    number=10_000,
    tries=5,
    min_speed_up=None,
    max_slow_down=None,
):
    aiperf_times = [
        timeit.timeit(aiperf_logger_func, number=number) for _ in range(tries)
    ]
    standard_times = [
        timeit.timeit(standard_logger_func, number=number) for _ in range(tries)
    ]

    aiperf_avg_time = sum(aiperf_times) / tries
    standard_avg_time = sum(standard_times) / tries
    slow_down = aiperf_avg_time / standard_avg_time
    speed_up = standard_avg_time / aiperf_avg_time
    func_name_aiperf = aiperf_logger_func.__name__
    func_name_standard = standard_logger_func.__name__

    print(
        f"AIPerf logger time: {aiperf_avg_time:.5f} seconds (min: {min(aiperf_times):.5f}, max: {max(aiperf_times):.5f})"
    )
    print(
        f"Standard logger time: {standard_avg_time:.5f} seconds (min: {min(standard_times):.5f}, max: {max(standard_times):.5f})"
    )

    slow_down_msg = f"AIPerf logger is {slow_down:.2f}x slower than standard logger for {func_name_aiperf} vs {func_name_standard} (expected at most {max_slow_down or 1 / (min_speed_up or 1):.2f}x)"
    speed_up_msg = f"AIPerf logger is {speed_up:.2f}x faster than standard logger for {func_name_aiperf} vs {func_name_standard} (expected at least {min_speed_up or 1 / (max_slow_down or 1):.2f}x)"

    if slow_down > 1:
        print(slow_down_msg)
    else:
        print(speed_up_msg)

    if max_slow_down is not None:
        assert slow_down <= max_slow_down, slow_down_msg
    if min_speed_up is not None:
        assert speed_up >= min_speed_up, speed_up_msg


class MockLogCall:
    """Simple class to store the arguments of a log call, with lazy evaluation support for the message argument."""

    def __init__(self):
        self.level = None
        self.msg = None
        self.args = None
        self.kwargs = None

    def __call__(self, level: int, msg: str | Callable[..., str], *args, **kwargs):
        self.level = level
        self.msg = msg() if callable(msg) else msg
        self.args = args
        self.kwargs = kwargs

    def reset(self):
        self.level = None
        self.msg = None
        self.args = None
        self.kwargs = None

    def assert_called_with(self, level: int, msg: str, *args, **kwargs):
        assert self.level == level
        assert self.msg == msg
        assert self.args == args
        assert self.kwargs == kwargs


class TestAIPerfLogger:
    def test_logger_initialization(self):
        """Test that logger initializes correctly."""
        logger = AIPerfLogger("test")
        assert logger._logger.name == "test"

    def test_exception_logging(self):
        """Test exception logging includes exc_info."""
        logger = AIPerfLogger("test")
        logger._log = Mock()
        logger.set_level(_ERROR)

        logger.exception("Error occurred")

        logger._log.assert_called_once_with(_ERROR, "Error occurred", exc_info=True)

    @pytest.mark.parametrize(
        "level,expected",
        [
            # Real levels
            ("DEBUG", True),
            ("INFO", True),
            ("WARNING", True),
            ("ERROR", True),
            ("CRITICAL", True),
            (_DEBUG, True),
            (_INFO, True),
            (_WARNING, True),
            (_ERROR, True),
            (_CRITICAL, True),
            # Custom levels
            (_TRACE, True),
            (_NOTICE, True),
            (_SUCCESS, True),
            ("TRACE", True),
            ("NOTICE", True),
            ("SUCCESS", True),
            # Fake levels
            ("INVALID", False),
            (999, False),
            (-1, False),
            ("SUPER_SECRET_LEVEL", False),
        ],
    )
    def test_is_valid_level(self, level, expected):
        """Test level validation."""
        assert AIPerfLogger.is_valid_level(level) == expected

    def test_legacy_methods(self):
        """Test that legacy methods delegate to the logger."""
        logger = AIPerfLogger("test")
        logger.setLevel(_WARNING)
        assert logger.getEffectiveLevel() == _WARNING
        assert logger.isEnabledFor(_WARNING)

    def test_trace_or_debug(self, time_traveler: TimeTraveler):
        """Test that the lambda overloading of trace_or_debug logs the correct message depending on the level of the logger."""
        logger = AIPerfLogger("test")
        logger.setLevel(_TRACE)

        mock_log = MockLogCall()
        logger._log = mock_log

        logger.trace_or_debug(
            lambda: f"Current time ns: {time_traveler.time_ns()}",
            lambda: f"Current perf counter ns: {time_traveler.perf_counter_ns()}",
        )
        mock_log.assert_called_with(
            _TRACE, f"Current time ns: {time_traveler.time_ns()}"
        )

        logger.setLevel(_DEBUG)
        logger.trace_or_debug(
            lambda: f"Current time ns: {time_traveler.time_ns()}",
            lambda: f"Current perf counter ns: {time_traveler.perf_counter_ns()}",
        )
        mock_log.assert_called_with(
            _DEBUG, f"Current perf counter ns: {time_traveler.perf_counter_ns()}"
        )

    def test_trace_or_debug_formatting(self):
        """Test that the format overloading of trace_or_debug logs the correct message depending on the level of the logger."""
        logger = AIPerfLogger("test")
        logger.setLevel(_TRACE)
        mock_log = MockLogCall()
        logger._log = mock_log

        monday = "Monday"
        tuesday = "Tuesday"

        logger.trace_or_debug(
            "The current day is: {}",
            monday,
            tuesday,
        )
        mock_log.assert_called_with(_TRACE, "The current day is: {}".format(monday))  # noqa: UP032

        mock_log.reset()
        logger.setLevel(_DEBUG)
        logger.trace_or_debug(
            "The current day is: {}",
            monday,
            tuesday,
        )
        mock_log.assert_called_with(_DEBUG, "The current day is: {}".format(tuesday))  # noqa: UP032

    def test_trace_or_debug_multiple_format_args(self):
        """Test that the format overloading of trace_or_debug logs the correct message depending on the level of the logger
        when multiple format arguments are provided."""
        logger = AIPerfLogger("test")
        logger.setLevel(_TRACE)
        mock_log = MockLogCall()
        logger._log = mock_log

        trace_args = ("Monday", "Tuesday")
        debug_args = ("Wednesday", "Thursday")

        logger.trace_or_debug(
            "The current day is: {} not {}",
            trace_args,
            debug_args,
        )
        mock_log.assert_called_with(
            _TRACE, "The current day is: {} not {}".format(*trace_args)
        )  # noqa: UP032

        mock_log.reset()
        logger.setLevel(_DEBUG)
        logger.trace_or_debug(
            "The current day is: {} not {}",
            trace_args,
            debug_args,
        )
        mock_log.assert_called_with(
            _DEBUG, "The current day is: {} not {}".format(*debug_args)
        )  # noqa: UP032

    def test_trace_or_debug_format_with_single_arg_tuple(self):
        """Test that the format overloading of trace_or_debug logs the correct message depending on the level of the logger
        when a single argument tuple is provided."""
        logger = AIPerfLogger("test")
        logger.setLevel(_TRACE)
        mock_log = MockLogCall()
        logger._log = mock_log

        trace_args = ("SAT",)
        debug_args = ("SUN",)

        logger.trace_or_debug(
            "The current day is: {}",
            trace_args,
            debug_args,
        )
        mock_log.assert_called_with(
            _TRACE, "The current day is: {}".format(*trace_args)
        )  # noqa: UP032

        mock_log.reset()
        logger.setLevel(_DEBUG)
        logger.trace_or_debug(
            "The current day is: {}",
            trace_args,
            debug_args,
        )
        mock_log.assert_called_with(
            _DEBUG, "The current day is: {}".format(*debug_args)
        )  # noqa: UP032


@pytest.mark.performance
class TestAIPerfLoggerPerformance:
    def test_aiperf_logger_with_lazy_evaluation_debug(
        self, aiperf_logger, standard_logger
    ):
        """
        Tests that the AIPerf logger is faster than the standard logger when lazy evaluation is used for
        f-string formatting when the log will NOT be printed.
        """

        def aiperf_lazy_f_string():
            aiperf_logger.debug(lambda: f"Hello, world! {time.time_ns() ** 2}")

        def standard_f_string():
            standard_logger.debug(f"Hello, world! {time.time_ns() ** 2}")

        # Expected to be faster than standard logger
        compare_logger_performance(
            aiperf_lazy_f_string,
            standard_f_string,
            min_speed_up=1.5,
        )

    def test_plain_string_debug(self, aiperf_logger, standard_logger):
        """
        Tests that the AIPerf logger is not a lot slower than the standard logger when a plain string is used for
        logging both with and without lazy evaluation when the log will NOT be printed.
        """

        def aiperf_plain_string():
            aiperf_logger.debug(
                "Hello, world! This is a test of an example message that will NOT be printed."
            )

        def aiperf_plain_string_lazy():
            aiperf_logger.debug(
                lambda: "Hello, world! This is a test of an example message that will NOT be printed."
            )

        def standard_plain_string():
            standard_logger.debug(
                "Hello, world! This is a test of an example message that will NOT be printed."
            )

        # Expected to be marginally slower than standard logger
        compare_logger_performance(
            aiperf_plain_string,
            standard_plain_string,
            max_slow_down=1.5,
        )

        # Expected to be marginally slower than standard logger
        compare_logger_performance(
            aiperf_plain_string_lazy,
            standard_plain_string,
            max_slow_down=1.5,
        )

    def test_plain_string_info(self, aiperf_logger, standard_logger):
        """
        Tests that the AIPerf logger is not a lot slower than the standard logger when a plain string is used for
        logging both with and without lazy evaluation when the log will be printed.
        """

        def aiperf_plain_string():
            aiperf_logger.info(
                "Hello, world! This is a test of an example message that will be printed."
            )

        def aiperf_plain_string_lazy():
            aiperf_logger.info(
                lambda: "Hello, world! This is a test of an example message that will be printed."
            )

        def standard_plain_string():
            standard_logger.info(
                "Hello, world! This is a test of an example message that will be printed."
            )

        # Expected to be marginally slower than standard logger
        compare_logger_performance(
            aiperf_plain_string,
            standard_plain_string,
            max_slow_down=1.5,
        )

        # Expected to be marginally slower than standard logger
        compare_logger_performance(
            aiperf_plain_string_lazy,
            standard_plain_string,
            max_slow_down=1.5,
        )

    def test_formatting_info(self, aiperf_logger, standard_logger):
        """
        Tests that the AIPerf logger is not a lot slower than the standard logger when the message is formatted
        using %s and will be printed.
        """

        def aiperf_formatting():
            aiperf_logger.info(
                "Hello, world! This will be printed %s " * 100, *["test"] * 100
            )

        def standard_formatting():
            standard_logger.info(
                "Hello, world! This will be printed %s" * 100, *["test"] * 100
            )

        # Expected to be marginally slower than standard logger
        compare_logger_performance(
            aiperf_formatting,
            standard_formatting,
            max_slow_down=1.5,
        )

    def test_lazy_evaluation_and_formatting_debug(self, aiperf_logger, standard_logger):
        """
        Tests that the AIPerf logger is faster than the standard logger when lazy evaluation is used for
        lazy %s formatting when the log will NOT be printed.
        """

        def aiperf_formatting_and_lazy_evaluation():
            aiperf_logger.debug(
                lambda: "Hello, world! This will NOT be printed %s "
                * 100
                % tuple([*["test"] * 100])
            )

        def standard_formatting_no_print():
            standard_logger.debug(
                "Hello, world! This will NOT be printed %s " * 100, *["test"] * 100
            )

        # Expected to be faster than standard logger
        compare_logger_performance(
            aiperf_formatting_and_lazy_evaluation,
            standard_formatting_no_print,
            min_speed_up=2,
        )

    def test_lazy_evaluation_and_formatting_and_multiple_args_debug(
        self, aiperf_logger, standard_logger
    ):
        """
        Tests that the AIPerf logger is not a lot slower than the standard logger when the message is formatted
        using %s and will NOT be printed.
        """

        def aiperf_multiple_args():
            aiperf_logger.debug(
                lambda: f"Hello Mr {time.time_ns() ** 2} {time.time_ns() ** 2} This will NOT be printed"
            )

        def standard_multiple_args():
            standard_logger.debug(
                "Hello Mr %d %d This will NOT be printed",
                time.time_ns() ** 2,
                time.time_ns() ** 2,
            )

        # Expected to be faster than standard logger
        compare_logger_performance(
            aiperf_multiple_args,
            standard_multiple_args,
            min_speed_up=2,
        )

    def test_message_formatting_info(self, aiperf_logger, standard_logger):
        """
        Tests that the AIPerf logger is not a lot slower than the standard logger when the message is formatted
        using %s and will be printed.
        """

        def aiperf_message_formatting():
            aiperf_logger.info(
                "Hello, world! This will be printed %s" * 100, *["test"] * 100
            )

        def standard_message_formatting():
            standard_logger.info(
                "Hello, world! This will be printed %s" * 100, *["test"] * 100
            )

        # Expected to be marginally slower than standard logger
        compare_logger_performance(
            aiperf_message_formatting,
            standard_message_formatting,
            max_slow_down=1.5,
        )

    def test_large_messages_debug(self, aiperf_logger, standard_logger, large_message):
        """
        Tests that the AIPerf logger is faster than the standard logger when lazy evaluation is used for
        f-string formatting large messages when the log will NOT be printed.
        """

        def aiperf_f_string_message():
            aiperf_logger.debug(lambda: f"Got message: {large_message}")

        def standard_f_string_message():
            standard_logger.debug(f"Got message: {large_message}")

        def standard_fmt_message():
            standard_logger.debug("Got message: %s", large_message)

        # Expected to be incredibly fast
        compare_logger_performance(
            aiperf_f_string_message,
            standard_f_string_message,
            min_speed_up=10,
        )

        # Expected to be marginally slower than standard logger
        compare_logger_performance(
            aiperf_f_string_message,
            standard_fmt_message,
            max_slow_down=1.5,
        )

    def test_large_messages_debug_math(
        self, aiperf_logger, standard_logger, large_message
    ):
        """
        Tests that the AIPerf logger is faster than the standard logger when lazy evaluation is used for simple math
        when the log will NOT be printed.
        """

        def aiperf_f_string_math():
            aiperf_logger.debug(
                lambda: f"Request time: {(large_message.end_perf_ns - large_message.start_perf_ns) / NANOS_PER_SECOND:.2f}"
            )

        def standard_f_string_math():
            standard_logger.debug(
                f"Request time: {(large_message.end_perf_ns - large_message.start_perf_ns) / NANOS_PER_SECOND:.2f}"
            )

        def standard_fmt_math():
            standard_logger.debug(
                "Request time: %.2f",
                (large_message.end_perf_ns - large_message.start_perf_ns)
                / NANOS_PER_SECOND,
            )

        # Expected to be decently faster
        compare_logger_performance(
            aiperf_f_string_math,
            standard_f_string_math,
            min_speed_up=1.5,
        )

        # Tests actually show this as being faster for AIPerf logger
        compare_logger_performance(
            aiperf_f_string_math,
            standard_fmt_math,
            max_slow_down=1.5,
        )
