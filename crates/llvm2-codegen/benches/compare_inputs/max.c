// max.c - Conditional max of two values
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

#include <stdint.h>

int64_t max(int64_t a, int64_t b) {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}
