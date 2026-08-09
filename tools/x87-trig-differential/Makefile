CXX ?= c++
CC ?= cc
CXXFLAGS ?= -O3 -std=c++20 -Wall -Wextra -Wpedantic
CFLAGS ?= -O3 -std=gnu11 -Wall -Wextra

.PHONY: all check check-v86 clean captured

all: x87-trig-diff

core_math_sincosf.o: core_math_sincosf.c
	$(CC) $(CFLAGS) -ffp-contract=off -c $< -o $@

x87-trig-diff.o: x87_trig_diff.cpp
	$(CXX) $(CXXFLAGS) -ffp-contract=off -c $< -o $@

v86-libm/target/release/libv86_libm_candidate.a: v86-libm/Cargo.toml v86-libm/src/lib.rs
	cargo build --manifest-path v86-libm/Cargo.toml --release

x87-trig-diff: x87-trig-diff.o core_math_sincosf.o v86-libm/target/release/libv86_libm_candidate.a
	$(CXX) $^ -ldl -lpthread -lm -o $@

check: x87-trig-diff
	./x87-trig-diff --random 10000 --deduplicate --samples 4 > /tmp/irisu-x87-trig-check.json
	python3 test_report.py /tmp/irisu-x87-trig-check.json

check-v86:
	python3 run_v86_probe.py

captured: x87-trig-diff
	./x87-trig-diff \
	  --input ../../.tmp/exact-core-perf-20260721/trig-keys-3000.bin \
	  --no-curated --deduplicate > /tmp/irisu-x87-trig-captured.json

clean:
	$(RM) x87-trig-diff x87-trig-diff.o core_math_sincosf.o
	cargo clean --manifest-path v86-libm/Cargo.toml
