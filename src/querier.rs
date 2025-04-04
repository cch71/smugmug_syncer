/*
 * Copyright (c) 2025 Craig Hamilton and Contributors.
 * Licensed under either of
 *  - Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> OR
 *  - MIT license <http://opensource.org/licenses/MIT>
 *  at your option.
 */

use crate::{QueryArgs, local_folder::LocalFolder};
use anyhow::Result;
use humansize::{DECIMAL, format_size};

pub(crate) async fn handle_query_req(path: &str, args: QueryArgs) -> Result<()> {
    println!("Querying {:#?}", args);
    let local_folder = LocalFolder::get(path)?;
    if args.print_tree {
        println!("SmugMug Folder Tree:\n{}", &local_folder);
    }

    if args.calc_size {
        let stats = local_folder.get_smugmug_folder_stats();
        println!(
            "Total Size of SmugMug folder: {}",
            format_size(stats.total_data_usage_in_bytes, DECIMAL)
        );
        println!(
            "Total Size of originals in SmugMug folder: {}",
            format_size(stats.original_data_usage_in_bytes, DECIMAL)
        );

        println!(
            "Total Size of unique images in SmugMug folder: {}",
            format_size(stats.unique_image_data_usage_in_bytes, DECIMAL)
        );
        println!("Total num of images: {}", stats.total_num_images);
        println!(
            "Total num of unique images: {}",
            stats.total_num_unique_images
        );
    }
    Ok(())
}
