# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 21:10:57 2024

@author: Shahir
"""

import os
from dataclasses import dataclass
from typing import Self, TypedDict, Any

import ujson as json

from alto_lib.utils import get_rel_pkg_path


class ClientLogAnalyzer:
    @dataclass(kw_only=True, frozen=True)
    class AnalysisOutput:
        start_time: float
        end_time: float
        time_length: float
        latency_avg: float
        latency_median: float
        latency_p99: float
        achieved_rate: float
        throughput_avg: float

    class _LogEntry(TypedDict):
        time: float
        last_request_sent_time: float | None
        last_response_recv_time: float | None
        avg_latency: float | None
        median_latency: float | None
        p99_latency: float | None
        num_requests_sent: int
        num_responses_recv: int

    _log_fname: str
    _log_metadata: dict[str, Any]
    _log_data: list[_LogEntry]

    def __init__(
        self: Self,
        log_fname: str
    ) -> None:

        self._log_fname = log_fname
        with open(self._log_fname, 'r') as f:
            data = f.read().split('\n')[:-1]
            log_lines = [json.loads(n) for n in data]
        self._log_metadata = log_lines[0]
        for n, entry in enumerate(log_lines):
            if 'avg_latency' in entry:
                break
        self._log_data = log_lines[n:]

    def get_first_nonzero_entry(
        self: Self
    ) -> _LogEntry | None:

        for entry in self._log_data:
            if entry['num_responses_recv'] > 0:
                return entry
        return None

    def get_last_increase_entry(
        self: Self
    ) -> _LogEntry | None:

        last_entry = None
        last_num_responses_recv = 0
        for entry in self._log_data:
            num_responses_recv = entry['num_requests_sent']
            if num_responses_recv > 0 and num_responses_recv > last_num_responses_recv:
                last_entry = entry
                last_num_responses_recv = num_responses_recv
        return last_entry

    def get_entry_after_time(
        self: Self,
        time_val: float
    ) -> _LogEntry | None:

        for entry in self._log_data:
            if entry['time'] >= time_val:
                return entry
        return None

    def get_entry_before_time(
        self: Self,
        time_val: float
    ) -> _LogEntry | None:

        last_entry = None
        for entry in self._log_data:
            if entry['time'] <= time_val:
                last_entry = entry
        return last_entry

    def get_test_stats(
        self: Self,
        test_length: float | None = None,
        start_time_offset: float = 0,
        end_time_offset: float = 0,
        start_time_nonzero: bool = True,
        end_time_increase: bool = True
    ) -> AnalysisOutput:

        if start_time_nonzero:
            first_entry = self.get_first_nonzero_entry()
            assert first_entry is not None
        else:
            first_entry = self._log_data[0]
        if start_time_offset != 0:
            first_entry = self.get_entry_after_time(first_entry['time'] + start_time_offset)
            assert first_entry is not None

        if test_length is not None:
            assert not end_time_increase
            last_entry = self.get_entry_after_time(first_entry['time'] + test_length)
            assert last_entry is not None
        else:
            if end_time_increase:
                last_entry = self.get_last_increase_entry()
                assert last_entry is not None
            else:
                last_entry = self._log_data[-1]
        if end_time_offset != 0:
            assert test_length is None
            last_entry = self.get_entry_before_time(last_entry['time'] + end_time_offset)

        assert not ((first_entry is None) or (last_entry is None))
        
        delta_requests_sent = last_entry['num_requests_sent'] - first_entry['num_requests_sent']
        delta_responses_recv = last_entry['num_responses_recv'] - first_entry['num_responses_recv']
        latency_avg = last_entry['avg_latency']
        latency_median = last_entry['median_latency']
        latency_p99 = last_entry['p99_latency']
        delta_time = last_entry['time'] - first_entry['time']
        achieved_rate = delta_requests_sent / delta_time
        throughput_avg = delta_responses_recv / delta_time

        assert latency_avg is not None
        assert latency_median is not None
        assert latency_p99 is not None

        out = self.AnalysisOutput(
            start_time=first_entry['time'],
            end_time=last_entry['time'],
            time_length=delta_time,
            latency_avg=latency_avg,
            latency_median=latency_median,
            latency_p99=latency_p99,
            achieved_rate=achieved_rate,
            throughput_avg=throughput_avg
        )

        return out


def write_log_data(
    log_dirname: str,
    output_fname: str
) -> None:

    fnames = os.listdir(log_dirname)
    analysis_datas: list[tuple[float, ClientLogAnalyzer.AnalysisOutput]] = []
    for fname in fnames:
        analyzer = ClientLogAnalyzer(os.path.join(log_dirname, fname))
        analysis_data = analyzer.get_test_stats(
            start_time_offset=30,
            end_time_offset=-60,
            start_time_nonzero=True,
            end_time_increase=True
        )
        rate = float(fname.split('_')[3])
        analysis_datas.append((rate, analysis_data))
    analysis_datas.sort(key=lambda x: x[0])
    
    csv_out = [
        "rate,achieved_rate,avg_latency,median_latency,p99_latency,throughput_avg\n"
    ]
    for rate, analysis_data in analysis_datas:
        line_data = [
            rate,
            analysis_data.achieved_rate,
            analysis_data.latency_avg,
            analysis_data.latency_median,
            analysis_data.latency_p99,
            analysis_data.throughput_avg
        ]
        csv_out.append(",".join(map(str, line_data)) + "\n")
    
    with open(output_fname, 'w') as f:
        f.writelines(csv_out)

if __name__ == '__main__':
    write_log_data(get_rel_pkg_path("logs"), "log_data.csv")