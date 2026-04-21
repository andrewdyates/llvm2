// max.c - Conditional max of two values
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

#include <stdint.h>

int64_t max(int64_t a, int64_t b) {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}
