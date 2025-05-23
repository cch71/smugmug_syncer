/*
 * Copyright (c) 2025 Craig Hamilton and Contributors.
 * Licensed under either of
 *  - Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> OR
 *  - MIT license <http://opensource.org/licenses/MIT>
 *  at your option.
 */

#pragma once

#include <iostream>
#include <mutex>
#include <optional>

template <typename T>
class OnceLock
{
public:
    template <typename F, typename... Args>
    T &get_or_init(F &&f, Args &&...args)
    {
        std::call_once(flag_, [&]()
                       { value_.emplace(std::forward<F>(f)(std::forward<Args>(args)...)); });
        return *value_;
    }

    bool has_value() const
    {
        return value_.has_value();
    }

    const T *get() const
    {
        return value_.has_value() ? &*value_ : nullptr;
    }

    T *get_mut()
    {
        return value_.has_value() ? &*value_ : nullptr;
    }

private:
    std::once_flag flag_;
    std::optional<T> value_;
};
