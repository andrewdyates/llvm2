// fibonacci.c - Iterative fibonacci with two accumulators
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

#include <stdint.h>

int64_t fibonacci(int64_t n) {
    int64_t a = 0;
    int64_t b = 1;
    int64_t i = 0;
    while (i < n) {
        int64_t tmp = a + b;
        a = b;
        b = tmp;
        i = i + 1;
    }
    return a;
}
