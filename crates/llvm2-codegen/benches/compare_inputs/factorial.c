// factorial.c - Iterative factorial with multiply loop
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

#include <stdint.h>

int64_t factorial(int64_t n) {
    int64_t result = 1;
    int64_t i = 2;
    while (i <= n) {
        result = result * i;
        i = i + 1;
    }
    return result;
}
