/*
 * Copyright (c) 2025 Craig Hamilton and Contributors.
 * Licensed under either of
 *  - Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> OR
 *  - MIT license <http://opensource.org/licenses/MIT>
 *  at your option.
 */

use smugmug::v2::{OAuth1Token, oauth1_access_token, oauth1_authorize_url, oauth1_request_token};
use std::fs::File;
use std::io::{self, BufReader};

// Retrieve smugmug tokens from the Oauth1 service
async fn get_smugmug_tokens_from_service(
    consumer_key: &str,
    consumer_secret: &str,
) -> anyhow::Result<OAuth1Token> {
    // step 1: acquire request token & token secret
    let request_token = oauth1_request_token(consumer_key, consumer_secret, "oob").await?;

    let auth_url = oauth1_authorize_url(&request_token, "Full", "Modify")?;
    println!("please access: {}", auth_url.as_str());

    // step 2. acquire user pin
    println!("input pin: ");
    let mut user_input = String::new();
    io::stdin().read_line(&mut user_input)?;
    let pin = user_input.trim();

    // step 3. acquire access token
    let access_token =
        oauth1_access_token(consumer_key, consumer_secret, &request_token, pin).await?;
    println!(
        "your token and secret is: \n token: {}\n secret: {}",
        access_token.token, access_token.secret
    );

    Ok(access_token)
}

// Retrieve oauth tokens from file
fn get_smugmug_tokens_from_file(file: File) -> anyhow::Result<OAuth1Token> {
    let reader = BufReader::new(file);
    Ok(serde_json::from_reader(reader)?)
}

// Get Consumer and Access Tokens
#[allow(dead_code)]
pub(crate) async fn get_full_auth_tokens() -> anyhow::Result<smugmug::v2::Creds> {
    let api_key = std::env::var("SMUGMUG_API_KEY")?;
    let api_secret = std::env::var("SMUGMUG_API_SECRET")?;
    let token_cache = std::env::var("SMUGMUG_AUTH_CACHE")?;

    let tokens = match File::open(token_cache.clone()) {
        Ok(file) => get_smugmug_tokens_from_file(file)?,
        Err(_) => {
            let tokens = get_smugmug_tokens_from_service(&api_key, &api_secret).await?;
            let token_str = serde_json::to_string(&tokens)?;
            std::fs::write(token_cache, &token_str)?;
            tokens
        }
    };

    Ok(smugmug::v2::Creds::from_tokens(
        &api_key,
        Some(&api_secret),
        Some(&tokens.token),
        Some(&tokens.secret),
    ))
}

// Get Consumer API Token only
#[allow(dead_code)]
pub(crate) async fn get_read_only_auth_tokens() -> anyhow::Result<smugmug::v2::Creds> {
    let api_key = std::env::var("SMUGMUG_API_KEY")?;

    Ok(smugmug::v2::Creds::from_tokens(&api_key, None, None, None))
}
