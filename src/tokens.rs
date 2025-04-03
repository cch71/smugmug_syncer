/*
 * Copyright (c) 2025 Craig Hamilton and Contributors.
 * Licensed under either of
 *  - Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> OR
 *  - MIT license <http://opensource.org/licenses/MIT>
 *  at your option.
 */

use const_format::concatcp;
use reqwest_oauth1::{OAuthClientProvider, TokenReaderFuture};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, BufReader};
use url::Url;

#[derive(Deserialize, Serialize, Debug)]
struct SmugMugOauth1Token {
    token: String,
    secret: String,
}

// Retrieve smugmug tokens from the Oauth1 service
async fn get_smugmug_tokens_from_service(
    consumer_key: &str,
    consumer_secret: &str,
) -> anyhow::Result<SmugMugOauth1Token> {
    const OAUTH_ORIGIN: &str = "https://secure.smugmug.com";
    const REQUEST_TOKEN_URL: &str =
        concatcp!(OAUTH_ORIGIN, "/services/oauth/1.0a/getRequestToken/");
    const ACCESS_TOKEN_URL: &str = concatcp!(OAUTH_ORIGIN, "/services/oauth/1.0a/getAccessToken/");
    const AUTHORIZE_URL: &str = concatcp!(OAUTH_ORIGIN, "/services/oauth/1.0a/authorize/");

    // step 1: acquire request token & token secret
    let secrets = reqwest_oauth1::Secrets::new(consumer_key, consumer_secret);

    let client = reqwest::Client::new();
    let resp = client
        .oauth1(secrets)
        .post(REQUEST_TOKEN_URL)
        .query(&[("oauth_callback", "oob")])
        .send()
        .parse_oauth_token()
        .await?;

    let rt = resp.oauth_token;
    let rts = resp.oauth_token_secret;
    let auth_url = Url::parse_with_params(
        AUTHORIZE_URL,
        &[
            ("oauth_token", rt.as_str()),
            ("access", "Full"),
            ("permissions", "Modify"),
        ],
    )?;
    println!("please access: {}", auth_url.as_str());

    // step 2. acquire user pin
    println!("input pin: ");
    let mut user_input = String::new();
    io::stdin().read_line(&mut user_input)?;
    let pin = user_input.trim();

    // step 3. acquire access token
    let secrets = reqwest_oauth1::Secrets::new(consumer_key, consumer_secret).token(rt, rts);

    let client = reqwest::Client::new();
    let resp = client
        .oauth1(secrets)
        .get(ACCESS_TOKEN_URL)
        .query(&[("oauth_verifier", pin)])
        .send()
        .parse_oauth_token()
        .await?;
    println!(
        "your token and secret is: \n token: {}\n secret: {}",
        resp.oauth_token, resp.oauth_token_secret
    );

    Ok(SmugMugOauth1Token {
        token: resp.oauth_token,
        secret: resp.oauth_token_secret,
    })
}

// Retrieve oauth tokens from file
fn get_smugmug_tokens_from_file(file: File) -> anyhow::Result<SmugMugOauth1Token> {
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
