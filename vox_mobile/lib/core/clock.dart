/// Injectable time source so time-dependent logic is deterministic in tests.
typedef Clock = DateTime Function();

DateTime systemClock() => DateTime.now();
