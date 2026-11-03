/*
 * Copyright (c) 2025 Craig Hamilton and Contributors.
 * Licensed under either of
 *  - Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> OR
 *  - MIT license <http://opensource.org/licenses/MIT>
 *  at your option.
 */

//! Retrying of SmugMug requests that can fail transiently

use std::time::Duration;

use anyhow::Result;

const MAX_ATTEMPTS: u32 = 3;

/// Runs the attempt up to 3 times, waiting 500ms between tries, and returns the first success
/// or the last error.  `what` describes the operation in the retry warning.
pub(crate) async fn with_retries<T, Fut>(what: &str, mut attempt: impl FnMut() -> Fut) -> Result<T>
where
    Fut: Future<Output = Result<T>>,
{
    let mut retries = 0;
    loop {
        match attempt().await {
            Ok(v) => return Ok(v),
            Err(err) => {
                retries += 1;
                if retries >= MAX_ATTEMPTS {
                    return Err(err);
                }
                log::warn!("{} retry #: {}", what, retries);
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}
