//! Bitcoin Chain Graph - CLI Application
//!
//! High-performance Bitcoin blockchain ingestion into Neo4j with checkpoint management.
//!
//! Commands:
//! - init-schema: Initialize Neo4j schema and create initial checkpoint
//! - ingest: Start fresh ingestion from genesis block
//! - resume: Resume ingestion from last checkpoint
//! - status: Display checkpoint status and progress
//!
//! Example usage:
//! ```bash
//! # Initialize schema
//! cargo run -- init-schema --config config/default.toml
//!
//! # Start ingestion
//! cargo run -- ingest --config config/default.toml
//!
//! # Resume after interruption
//! cargo run -- resume --config config/default.toml
//!
//! # Check status
//! cargo run -- status --config config/default.toml
//! ```

use anyhow::{Context, Result};
use bitcoin::Network;
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use bitcoin_chain_graph::config::{Config, ConfigLoader};
use bitcoin_chain_graph::domain::{IngestionOrchestrator, ShutdownHeightTracker};
use bitcoin_chain_graph::parser::{RpcBlockProvider, SingleBlockLoader, ZmqBlockListener};
use bitcoin_chain_graph::writer::{CsvWriter, GraphWriter, Neo4jWriter, WriterError};

/// Bitcoin Chain Graph - Blockchain ingestion into Neo4j
#[derive(Parser)]
#[command(name = "bitcoin-chain-graph")]
#[command(about = "High-performance Bitcoin blockchain ingestion into Neo4j", long_about = None)]
#[command(version)]
struct Cli {
    /// Path to configuration file
    #[arg(
        short,
        long,
        global = true,
        value_name = "FILE",
        default_value = "config/default.toml"
    )]
    config: PathBuf,

    /// Writer type: "neo4j" or "csv" (default: "neo4j")
    #[arg(long, global = true, value_name = "TYPE", default_value = "neo4j")]
    writer: String,

    /// Output directory for CSV files (required when writer=csv)
    #[arg(long, global = true, value_name = "DIR")]
    csv_output_dir: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize Neo4j schema and create initial checkpoint
    InitSchema,

    /// Start fresh ingestion from genesis block (lazy streaming mode)
    Ingest {
        /// Maximum block height to process (default: all blocks)
        #[arg(long)]
        max_height: Option<u32>,
    },

    /// Resume ingestion from last checkpoint (lazy streaming with cache pre-warming)
    Resume {
        /// Maximum block height to process (default: all blocks)
        #[arg(long)]
        max_height: Option<u32>,
    },

    /// Display checkpoint status and progress
    Status,

    /// Live mode: catch up via RPC then stream new blocks in real-time via ZMQ
    Live {
        /// Maximum block height to process (default: follow chain tip indefinitely)
        #[arg(long)]
        max_height: Option<u32>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Load configuration
    let config = ConfigLoader::from_file(&cli.config)
        .with_context(|| format!("Failed to load config from {:?}", cli.config))?;

    config
        .validate()
        .context("Configuration validation failed")?;

    // Initialize logging
    init_logging(&config);

    // Validate writer type
    let writer_type = cli.writer.to_lowercase();
    if writer_type != "neo4j" && writer_type != "csv" {
        anyhow::bail!(
            "Invalid writer type: {}. Must be 'neo4j' or 'csv'",
            cli.writer
        );
    }

    // Validate CSV output directory if using CSV writer
    if writer_type == "csv" && cli.csv_output_dir.is_none() {
        anyhow::bail!("--csv-output-dir is required when --writer=csv");
    }

    match cli.command {
        Commands::InitSchema => {
            init_schema(&config, &writer_type, cli.csv_output_dir.as_deref()).await
        }
        Commands::Ingest { max_height } => {
            ingest(
                &config,
                max_height,
                &writer_type,
                cli.csv_output_dir.as_deref(),
            )
            .await
        }
        Commands::Resume { max_height } => {
            resume(
                &config,
                max_height,
                &writer_type,
                cli.csv_output_dir.as_deref(),
            )
            .await
        }
        Commands::Status => status(&config, &writer_type, cli.csv_output_dir.as_deref()).await,
        Commands::Live { max_height } => {
            run_live_ingestion(
                &config,
                max_height,
                &writer_type,
                cli.csv_output_dir.as_deref(),
            )
            .await
        }
    }
}

/// Initialize tracing subscriber for structured logging
fn init_logging(config: &Config) {
    // Create env filter: if RUST_LOG is set, use it directly for full control.
    // Otherwise, use config level but filter noisy HTTP client crates.
    let (env_filter, log_level) = if let Ok(rust_log) = std::env::var("RUST_LOG") {
        // RUST_LOG explicitly set — use as-is for full control
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        (filter, rust_log)
    } else {
        // Build a filter that quiets noisy HTTP client internals
        let app_level = &config.logging.level;
        let filter_str = format!(
            "{},hyper=warn,hyper_util=warn,reqwest=warn,h2=warn",
            app_level
        );
        let filter = EnvFilter::try_new(&filter_str).unwrap_or_else(|_| EnvFilter::new("info"));
        (filter, app_level.clone())
    };

    // Setup formatting layer
    let fmt_layer = fmt::layer()
        .with_target(true)
        .with_thread_ids(false)
        .with_line_number(false)
        .with_ansi(true);

    // Initialize subscriber
    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .init();

    tracing::info!(
        log_level = %log_level,
        config_file = ?config,
        "Logging initialized"
    );
}

/// Initialize schema and create initial checkpoint
async fn init_schema(
    config: &Config,
    writer_type: &str,
    csv_output_dir: Option<&Path>,
) -> Result<()> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    if writer_type == "csv" {
        println!("║  Initialize CSV Export Schema                                  ║");
    } else {
        println!("║  Initialize Neo4j Schema                                       ║");
    }
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let cache_size = config.performance.cache_capacity();

    match writer_type {
        "neo4j" => {
            println!("🔌 Connecting to Neo4j at {}...", config.neo4j.uri);
            let writer = Neo4jWriter::new(config.neo4j.clone())
                .await
                .context("Failed to connect to Neo4j")?;
            println!("   ✅ Connected successfully");

            let orchestrator = IngestionOrchestrator::new(writer, Network::Bitcoin, cache_size);

            println!("\n🏗️  Initializing schema...");
            orchestrator
                .init_schema()
                .await
                .context("Failed to initialize schema")?;
            println!("   ✅ Schema initialized");
        }
        "csv" => {
            let output_dir =
                csv_output_dir.context("CSV output directory is required when using CSV writer")?;
            println!("📁 Initializing CSV writer at {:?}...", output_dir);
            let writer = CsvWriter::new(output_dir)
                .await
                .context("Failed to create CSV writer")?;
            println!("   ✅ CSV writer initialized");

            let orchestrator = IngestionOrchestrator::new(writer, Network::Bitcoin, cache_size);

            println!("\n🏗️  Initializing schema...");
            orchestrator
                .init_schema()
                .await
                .context("Failed to initialize schema")?;
            println!("   ✅ Schema initialized");
        }
        _ => anyhow::bail!("Invalid writer type: {}", writer_type),
    }

    println!("\n✅ Initialization complete!");
    println!("\nNext steps:");
    println!("  cargo run -- ingest    # Start ingestion from genesis");
    println!("  cargo run -- status    # Check checkpoint status");

    Ok(())
}

/// Start fresh ingestion from genesis block (streaming mode)
async fn ingest(
    config: &Config,
    cli_max_height: Option<u32>,
    writer_type: &str,
    csv_output_dir: Option<&Path>,
) -> Result<()> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Start Fresh Ingestion (Streaming Mode)                       ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let cache_size = config.performance.cache_capacity();

    match writer_type {
        "neo4j" => {
            println!("🔌 Connecting to Neo4j at {}...", config.neo4j.uri);
            let writer = Neo4jWriter::new(config.neo4j.clone())
                .await
                .context("Failed to connect to Neo4j")?;
            println!("   ✅ Connected successfully");

            let orchestrator = IngestionOrchestrator::new(writer, Network::Bitcoin, cache_size)
                .with_max_transaction_memory_mb(config.ingestion.max_transaction_memory_mb);

            run_ingest_with_orchestrator(config, cli_max_height, orchestrator).await
        }
        "csv" => {
            let output_dir =
                csv_output_dir.context("CSV output directory is required when using CSV writer")?;
            println!("📁 Initializing CSV writer at {:?}...", output_dir);
            let writer = CsvWriter::new(output_dir)
                .await
                .context("Failed to create CSV writer")?;
            println!("   ✅ CSV writer initialized");

            let orchestrator = IngestionOrchestrator::new(writer, Network::Bitcoin, cache_size)
                .with_max_transaction_memory_mb(config.ingestion.max_transaction_memory_mb);

            run_ingest_with_orchestrator(config, cli_max_height, orchestrator).await
        }
        _ => anyhow::bail!("Invalid writer type: {}", writer_type),
    }
}

/// Common ingestion logic that works with any writer type
async fn run_ingest_with_orchestrator<W: GraphWriter + 'static>(
    config: &Config,
    cli_max_height: Option<u32>,
    orchestrator: IngestionOrchestrator<W>,
) -> Result<()> {
    // Check if schema is initialized
    let checkpoint = orchestrator
        .get_checkpoint()
        .await
        .context("Failed to check checkpoint")?;

    if checkpoint.is_none() {
        println!("\n⚠️  Warning: No checkpoint found. Initializing schema...");
        orchestrator
            .init_schema()
            .await
            .context("Failed to initialize schema")?;
        println!("   ✅ Schema initialized");
    }

    // Get resume height
    let resume_height = orchestrator
        .get_resume_height()
        .await
        .context("Failed to get resume height")?;

    if resume_height > 0 {
        println!(
            "\n⚠️  Warning: Existing ingestion found at block {}!",
            resume_height
        );
        println!("   Use 'resume' command to continue from last checkpoint");
        println!("   Or manually reset the checkpoint to start fresh");
        return Ok(());
    }

    // Determine max_height: CLI arg > config.bitcoin.end_height > chain tip
    let max_height = cli_max_height
        .or(config.bitcoin.end_height)
        .unwrap_or(u32::MAX); // Will stop when blocks run out

    // Create lazy-loading block loader (instant startup)
    println!("\n📚 Initializing lazy block loader (instant startup)...");
    let loader = SingleBlockLoader::new(&config.bitcoin.blocks_dir, Network::Bitcoin)
        .context("Failed to initialize block loader")?;

    println!(
        "   Target height range: 0 to {}",
        if max_height == u32::MAX {
            "chain tip".to_string()
        } else {
            max_height.to_string()
        }
    );

    // Start streaming ingestion (no pre-warming for fresh start)
    run_streaming_ingestion(config, orchestrator, loader, 0, max_height, false).await
}

/// Resume ingestion from last checkpoint (streaming with pre-warming)
async fn resume(
    config: &Config,
    cli_max_height: Option<u32>,
    writer_type: &str,
    csv_output_dir: Option<&Path>,
) -> Result<()> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Resume Ingestion (Streaming with Pre-warming)                ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let cache_size = config.performance.cache_capacity();

    match writer_type {
        "neo4j" => {
            println!("🔌 Connecting to Neo4j at {}...", config.neo4j.uri);
            let writer = Neo4jWriter::new(config.neo4j.clone())
                .await
                .context("Failed to connect to Neo4j")?;
            println!("   ✅ Connected successfully");

            let orchestrator = IngestionOrchestrator::new(writer, Network::Bitcoin, cache_size)
                .with_max_transaction_memory_mb(config.ingestion.max_transaction_memory_mb);

            run_resume_with_orchestrator(config, cli_max_height, orchestrator).await
        }
        "csv" => {
            let output_dir =
                csv_output_dir.context("CSV output directory is required when using CSV writer")?;
            println!("📁 Initializing CSV writer at {:?}...", output_dir);
            let writer = CsvWriter::new(output_dir)
                .await
                .context("Failed to create CSV writer")?;
            println!("   ✅ CSV writer initialized");

            let orchestrator = IngestionOrchestrator::new(writer, Network::Bitcoin, cache_size)
                .with_max_transaction_memory_mb(config.ingestion.max_transaction_memory_mb);

            run_resume_with_orchestrator(config, cli_max_height, orchestrator).await
        }
        _ => anyhow::bail!("Invalid writer type: {}", writer_type),
    }
}

/// Common resume logic that works with any writer type
async fn run_resume_with_orchestrator<W: GraphWriter + 'static>(
    config: &Config,
    cli_max_height: Option<u32>,
    orchestrator: IngestionOrchestrator<W>,
) -> Result<()> {
    // Get checkpoint
    let checkpoint = orchestrator
        .get_checkpoint()
        .await
        .context("Failed to get checkpoint")?
        .context("No checkpoint found. Run 'init-schema' first.")?;

    println!("\n📍 Checkpoint Status:");
    println!(
        "   Last processed height: {}",
        checkpoint.last_processed_height
    );
    println!("   Last processed hash: {}", checkpoint.last_processed_hash);
    println!("   Last processed file: {}", checkpoint.last_processed_file);
    println!("   Status: {}", checkpoint.status);

    if checkpoint.status == "completed" {
        println!("\n✅ Ingestion already complete!");
        return Ok(());
    }

    if checkpoint.status == "error" {
        println!("\n⚠️  Warning: Last ingestion ended with error status");
        println!("   Review logs and fix any issues before resuming");
        println!("   To resume anyway, the error status will be updated to 'in_progress'");
    }

    // Sync checkpoint with database state (handles crash recovery)
    // This detects blocks written but not checkpointed, verifies completeness,
    // and updates the checkpoint to preserve work from complete blocks.
    let resume_height = orchestrator
        .sync_checkpoint_with_db()
        .await
        .context("Failed to sync checkpoint with database")?;

    println!("\n🔄 Resume Plan:");
    println!("   Resume from block: {}", resume_height);

    // Determine max_height: CLI arg > config.bitcoin.end_height > chain tip
    let max_height = cli_max_height
        .or(config.bitcoin.end_height)
        .unwrap_or(u32::MAX); // Will stop when blocks run out

    // Create lazy-loading block loader (instant startup)
    tracing::info!("Initializing lazy block loader");
    let loader = SingleBlockLoader::new(&config.bitcoin.blocks_dir, Network::Bitcoin)
        .context("Failed to initialize block loader")?;

    tracing::info!(
        start = resume_height,
        end = if max_height == u32::MAX {
            "chain tip".to_string()
        } else {
            max_height.to_string()
        },
        "Target height range"
    );

    if resume_height >= max_height && max_height != u32::MAX {
        tracing::info!("No new blocks to process. Ingestion is up to date");
        return Ok(());
    }

    // Start streaming ingestion with pre-warming
    run_streaming_ingestion(
        config,
        orchestrator,
        loader,
        resume_height,
        max_height,
        true,
    )
    .await
}

/// Core streaming ingestion function with optional cache pre-warming
async fn run_streaming_ingestion<W: GraphWriter + 'static>(
    config: &Config,
    orchestrator: IngestionOrchestrator<W>,
    mut loader: SingleBlockLoader,
    start_height: u32,
    max_height: u32,
    enable_prewarm: bool,
) -> Result<()> {
    let start_time = Instant::now();
    let mut blocks_processed = 0;
    let cache_file = &config.performance.utxo_cache_file;

    // Wire up snapshot path so orchestrator saves after each committed batch
    let snapshot_path = if cache_file.is_empty() {
        None
    } else {
        Some(std::path::PathBuf::from(cache_file))
    };
    orchestrator.set_cache_snapshot_path(snapshot_path);

    // Try to load UTXO cache from snapshot file (faster than pre-warming from files)
    if !cache_file.is_empty() {
        let cache = orchestrator.get_cache();
        match cache.load_from_file(cache_file, Some(start_height)) {
            Ok(loaded) if loaded > 0 => {
                println!(
                    "   ✅ UTXO cache restored: {} entries ({:.1}% full)",
                    loaded,
                    cache.fill_percentage() * 100.0
                );
            }
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(error = %e, "Failed to load UTXO cache file, starting with empty cache");
            }
        }
    }

    // Pre-warm cache if resuming and configured
    if enable_prewarm && config.performance.utxo_prewarm_depth > 0 && start_height > 0 {
        let cache = orchestrator.get_cache();
        cache.enable_prewarm_mode();

        let prewarm_blocks = loader
            .prewarm_cache(cache, start_height, config.performance.utxo_prewarm_depth)
            .await
            .context("Cache pre-warming failed")?;

        cache.disable_prewarm_mode();

        println!(
            "   Pre-warmed {} blocks, cache at {:.1}%",
            prewarm_blocks,
            cache.fill_percentage() * 100.0
        );
    }

    // Pre-load full index range (single scan optimization)
    if max_height != u32::MAX {
        tracing::info!(
            "Pre-loading index for range {}-{}",
            start_height,
            max_height
        );
        loader
            .preload_full_range(start_height, max_height)
            .context("Failed to preload index range")?;
    }

    // Start streaming ingestion with batching
    // Fixed read-ahead size for disk I/O; adaptive memory splitting happens inside the orchestrator
    const READ_AHEAD_BLOCKS: usize = 5000;
    tracing::info!("Starting streaming ingestion with batching");
    let batch_size = READ_AHEAD_BLOCKS;
    tracing::info!(batch_size = batch_size, "Batch configuration");

    let mut batch: Vec<(u32, bitcoin::Block, String)> = Vec::with_capacity(batch_size);
    let mut batch_start_height = start_height;

    for height in start_height..=max_height {
        // Load single block (show progress during loading)
        if batch.is_empty() {
            batch_start_height = height;
            let end_range = (height + batch_size as u32 - 1).min(max_height);
            tracing::info!(start = height, end = end_range, "Loading batch");
        }

        // Show progress every 50 blocks during loading
        if height > batch_start_height && (height - batch_start_height) % 50 == 0 {
            let progress = height - batch_start_height;
            let total = batch_size.min((max_height - batch_start_height + 1) as usize);
            let pct = (progress as f64 / total as f64) * 100.0;
            tracing::debug!(
                progress = progress,
                total = total,
                percent = format!("{:.0}%", pct),
                "Batch loading progress"
            );
        }

        let (h, block, file_name) = match loader.load_block(height)? {
            Some(data) => data,
            None => {
                tracing::warn!(height = height, "Block not found, stopping ingestion");
                break;
            }
        };

        // Add to batch
        batch.push((h, block, file_name));

        // Ingest batch when full or at end
        if batch.len() >= batch_size || height == max_height {
            if batch.is_empty() {
                continue;
            }
            let batch_start = Instant::now();
            let batch_start_height = batch.first().expect("batch verified non-empty").0;
            let batch_end_height = batch.last().expect("batch verified non-empty").0;

            tracing::info!(
                start = batch_start_height,
                end = batch_end_height,
                "Ingesting batch"
            );

            match orchestrator.ingest_blocks_batch(&batch).await {
                Ok(_batch_stats) => {}
                Err(e) => {
                    let stats = orchestrator.cache_stats();
                    tracing::error!(
                        error = %e,
                        blocks_processed,
                        cache_hits = stats.hits,
                        cache_misses = stats.misses,
                        hit_rate_pct = format!("{:.2}", stats.hit_rate_percent()),
                        "Batch ingestion failed — cache stats at time of failure"
                    );
                    return Err(e).with_context(|| {
                        format!(
                            "Failed to ingest batch starting at block {}",
                            batch_start_height
                        )
                    });
                }
            }

            blocks_processed += batch.len();

            let elapsed = batch_start.elapsed().as_secs_f64();
            let bps = batch.len() as f64 / elapsed;

            let stats = orchestrator.cache_stats();
            let cache = orchestrator.get_cache();
            tracing::info!(
                blocks_per_sec = format!("{:.1}", bps),
                cache_size = orchestrator.cache_size(),
                cache_capacity = config.performance.cache_capacity(),
                cache_memory_mb = config.performance.utxo_cache_memory_mb,
                cache_fill_pct = format!("{:.1}", cache.fill_percentage() * 100.0),
                hit_rate_pct = format!("{:.1}", stats.hit_rate_percent()),
                "Batch complete"
            );

            // Clear batch for next iteration
            batch.clear();
        }
    }

    // Save cache on completion
    if !cache_file.is_empty() {
        let final_height = start_height + blocks_processed as u32;
        if let Err(e) = orchestrator
            .get_cache()
            .save_to_file(cache_file, final_height)
        {
            tracing::warn!(error = %e, "Failed to save UTXO cache on completion");
        }
    }

    let total_duration = start_time.elapsed();
    let overall_bps = blocks_processed as f64 / total_duration.as_secs_f64();

    // Show final cache stats
    let stats = orchestrator.cache_stats();

    tracing::info!(
        blocks_processed = blocks_processed,
        duration_secs = format!("{:.2}", total_duration.as_secs_f64()),
        avg_blocks_per_sec = format!("{:.2}", overall_bps),
        cache_size = orchestrator.cache_size(),
        cache_capacity = config.performance.cache_capacity(),
        cache_memory_mb = config.performance.utxo_cache_memory_mb,
        cache_fill_pct = format!(
            "{:.1}",
            orchestrator.cache_size() as f64 / config.performance.cache_capacity() as f64 * 100.0
        ),
        hit_rate_pct = format!("{:.2}", stats.hit_rate_percent()),
        hits = stats.hits,
        misses = stats.misses,
        "Streaming ingestion complete"
    );

    Ok(())
}

/// Display checkpoint status and progress
async fn status(config: &Config, writer_type: &str, csv_output_dir: Option<&Path>) -> Result<()> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Checkpoint Status                                             ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let cache_size = config.performance.cache_capacity();

    match writer_type {
        "neo4j" => {
            println!("🔌 Connecting to Neo4j at {}...", config.neo4j.uri);
            let writer = Neo4jWriter::new(config.neo4j.clone())
                .await
                .context("Failed to connect to Neo4j")?;
            let orchestrator = IngestionOrchestrator::new(writer, Network::Bitcoin, cache_size);
            run_status_with_orchestrator(orchestrator).await
        }
        "csv" => {
            let output_dir =
                csv_output_dir.context("CSV output directory is required when using CSV writer")?;
            println!("📁 Initializing CSV writer at {:?}...", output_dir);
            let writer = CsvWriter::new(output_dir)
                .await
                .context("Failed to create CSV writer")?;
            let orchestrator = IngestionOrchestrator::new(writer, Network::Bitcoin, cache_size);
            run_status_with_orchestrator(orchestrator).await
        }
        _ => anyhow::bail!("Invalid writer type: {}", writer_type),
    }
}

/// Common status logic that works with any writer type
async fn run_status_with_orchestrator<W: GraphWriter + 'static>(
    orchestrator: IngestionOrchestrator<W>,
) -> Result<()> {
    // Get checkpoint
    let checkpoint = orchestrator
        .get_checkpoint()
        .await
        .context("Failed to query checkpoint")?;

    match checkpoint {
        Some(cp) => {
            println!("📍 Current Checkpoint:");
            println!("   Status: {}", cp.status);
            println!("   Last processed height: {}", cp.last_processed_height);
            println!("   Last processed hash: {}", cp.last_processed_hash);
            println!("   Last processed file: {}", cp.last_processed_file);

            if let Some(offset) = cp.last_processed_file_offset {
                println!("   File offset: {} bytes", offset);
            }

            let timestamp = chrono::DateTime::from_timestamp(cp.timestamp, 0)
                .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
                .unwrap_or_else(|| "Unknown".to_string());
            println!("   Last updated: {}", timestamp);

            println!("\n🔄 Resume Information:");
            let resume_height = orchestrator.get_resume_height().await?;
            println!("   Next block to process: {}", resume_height);

            if cp.status == "in_progress" {
                println!("\n💡 Tip: Use 'resume' to continue ingestion");
            } else if cp.status == "completed" {
                println!("\n✅ Ingestion is complete!");
            } else if cp.status == "error" {
                println!("\n⚠️  Last ingestion ended with errors");
                println!("   Review logs and use 'resume' to retry");
            }
        }
        None => {
            println!("❌ No checkpoint found");
            println!("\n💡 Next steps:");
            println!("   1. Run: cargo run -- init-schema");
            println!("   2. Run: cargo run -- ingest");
        }
    }

    Ok(())
}

/// Live ingestion: catch up via RPC then stream new blocks via ZMQ
async fn run_live_ingestion(
    config: &Config,
    cli_max_height: Option<u32>,
    writer_type: &str,
    csv_output_dir: Option<&Path>,
) -> Result<()> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Live Mode: RPC Catchup + ZMQ Real-Time                      ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Validate RPC config
    config
        .validate_rpc()
        .context("RPC configuration validation failed for live mode")?;

    let cache_size = config.performance.cache_capacity();

    match writer_type {
        "neo4j" => {
            println!("🔌 Connecting to Neo4j at {}...", config.neo4j.uri);
            let writer = Neo4jWriter::new(config.neo4j.clone())
                .await
                .context("Failed to connect to Neo4j")?;
            println!("   ✅ Connected to Neo4j");

            let orchestrator = IngestionOrchestrator::new(writer, Network::Bitcoin, cache_size)
                .with_max_transaction_memory_mb(config.ingestion.max_transaction_memory_mb);

            run_live_with_orchestrator(config, cli_max_height, orchestrator).await
        }
        "csv" => {
            let output_dir =
                csv_output_dir.context("CSV output directory is required when using CSV writer")?;
            println!("📁 Initializing CSV writer at {:?}...", output_dir);
            let writer = CsvWriter::new(output_dir)
                .await
                .context("Failed to create CSV writer")?;
            println!("   ✅ CSV writer initialized");

            let orchestrator = IngestionOrchestrator::new(writer, Network::Bitcoin, cache_size)
                .with_max_transaction_memory_mb(config.ingestion.max_transaction_memory_mb);

            run_live_with_orchestrator(config, cli_max_height, orchestrator).await
        }
        _ => anyhow::bail!("Invalid writer type: {}", writer_type),
    }
}

/// Common live ingestion logic that works with any writer type
async fn run_live_with_orchestrator<W: GraphWriter + 'static>(
    config: &Config,
    cli_max_height: Option<u32>,
    orchestrator: IngestionOrchestrator<W>,
) -> Result<()> {
    // Validate RPC config
    let rpc_config = config
        .bitcoin_rpc
        .as_ref()
        .context("[bitcoin_rpc] section missing from config file. Required for live mode.")?;

    // Wire up snapshot path so orchestrator saves after each committed batch
    let cache_file = &config.performance.utxo_cache_file;
    let snapshot_path = if cache_file.is_empty() {
        None
    } else {
        Some(std::path::PathBuf::from(cache_file))
    };
    orchestrator.set_cache_snapshot_path(snapshot_path);

    // Try to load UTXO cache from snapshot file
    if !cache_file.is_empty() {
        let checkpoint_data = orchestrator.get_checkpoint().await.ok().flatten();
        let current_height = checkpoint_data.as_ref().and_then(|cp| {
            if cp.last_processed_height >= 0 {
                Some(cp.last_processed_height as u32)
            } else {
                None
            }
        });
        let cache = orchestrator.get_cache();
        match cache.load_from_file(cache_file, current_height) {
            Ok(loaded) if loaded > 0 => {
                println!(
                    "   ✅ UTXO cache restored: {} entries ({:.1}% full)",
                    loaded,
                    cache.fill_percentage() * 100.0
                );
            }
            Ok(_) => {} // No file or empty, will start cold
            Err(e) => {
                tracing::warn!(error = %e, "Failed to load UTXO cache file, starting with empty cache");
            }
        }
    }

    // Ensure schema is initialized
    let checkpoint = orchestrator
        .get_checkpoint()
        .await
        .context("Failed to check checkpoint")?;
    if checkpoint.is_none() {
        println!("\n🏗️  No checkpoint found. Initializing schema...");
        orchestrator
            .init_schema()
            .await
            .context("Failed to initialize schema")?;
        println!("   ✅ Schema initialized");
    }

    // Sync checkpoint with database state (handles crash recovery)
    // This detects blocks written but not checkpointed, verifies completeness,
    // and updates the checkpoint to preserve work from complete blocks.
    let resume_height = orchestrator
        .sync_checkpoint_with_db()
        .await
        .context("Failed to sync checkpoint with database")?;

    // Create RPC provider and verify connectivity
    println!(
        "\n🔌 Connecting to Bitcoin Core RPC at {}...",
        rpc_config.url
    );
    let provider = RpcBlockProvider::new(rpc_config).context("Failed to create RPC provider")?;

    let tip = provider.verify_connection().await?;
    println!("   ✅ Connected to Bitcoin Core (chain tip: {})", tip);

    let max_height = cli_max_height.unwrap_or(u32::MAX);
    let rpc_batch_size = provider.batch_size();

    let start_time = Instant::now();
    let mut current_height = resume_height;
    let mut blocks_processed: u64 = 0;
    let mut shutdown_tracker = ShutdownHeightTracker::new();

    println!("\n📊 Live Mode Configuration:");
    println!("   Resume from block: {}", resume_height);
    println!("   Chain tip: {}", tip);
    println!("   Blocks behind: {}", tip.saturating_sub(resume_height));
    println!("   RPC batch size: {}", rpc_batch_size);
    println!("   ZMQ endpoint: {}", rpc_config.zmq_endpoint);

    // Set up graceful shutdown via Ctrl+C (SIGINT) AND SIGTERM (systemd stop)
    let shutdown_token = CancellationToken::new();
    let token_clone = shutdown_token.clone();
    tokio::spawn(async move {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            let mut sigterm =
                signal(SignalKind::terminate()).expect("Failed to register SIGTERM handler");
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {
                    tracing::info!("SIGINT received, finishing current operation...");
                }
                _ = sigterm.recv() => {
                    tracing::info!("SIGTERM received, finishing current operation...");
                }
            }
        }
        #[cfg(not(unix))]
        {
            tokio::signal::ctrl_c().await.ok();
            tracing::info!("Shutdown signal received, finishing current operation...");
        }
        token_clone.cancel();
    });

    // ═══════════════════════════════════════════════════════════════
    // PHASE A: CATCHUP - Consume blocks as fast as possible via RPC
    // ═══════════════════════════════════════════════════════════════
    println!("\n🚀 Phase A: Catchup via RPC...");

    loop {
        if shutdown_token.is_cancelled() {
            tracing::info!("Shutdown requested during catchup phase");
            break;
        }

        // Re-check tip (it may advance during catchup)
        let tip = provider
            .get_tip_height()
            .await
            .context("Failed to get chain tip during catchup")?;
        let target = tip.min(max_height);

        if current_height > target {
            tracing::info!(current_height, chain_tip = tip, "Caught up to chain tip");
            break;
        }

        // Calculate batch size
        let remaining = (target - current_height + 1) as usize;
        let fetch_count = remaining.min(rpc_batch_size);

        tracing::debug!(
            start = current_height,
            count = fetch_count,
            tip = tip,
            behind = tip - current_height,
            "Fetching block batch via RPC"
        );

        // Fetch blocks in parallel
        let fetch_start = Instant::now();
        let blocks = provider
            .get_block_batch(current_height, fetch_count)
            .await
            .with_context(|| format!("Failed to fetch RPC batch at height {}", current_height))?;

        if blocks.is_empty() {
            tracing::warn!("Empty batch returned, stopping catchup");
            break;
        }

        // Check for shutdown after fetch completes
        if shutdown_token.is_cancelled() {
            tracing::info!("Shutdown requested after fetch, stopping catchup");
            break;
        }

        let fetch_elapsed = fetch_start.elapsed();
        let batch_end = blocks.last().map(|(h, _, _)| *h).unwrap_or(current_height);

        tracing::info!(
            fetched = blocks.len(),
            fetch_secs = format!("{:.2}", fetch_elapsed.as_secs_f64()),
            start = current_height,
            behind = tip - current_height,
            "RPC batch fetched"
        );

        // Ingest using existing orchestrator
        let ingest_start = Instant::now();
        let batch_result = orchestrator.ingest_blocks_batch(&blocks).await;
        let ingest_elapsed = ingest_start.elapsed();

        let ingestion_stats = match batch_result {
            Ok(stats) => stats,
            Err(e) => {
                let cache_stats = orchestrator.cache_stats();
                tracing::error!(
                    error = %e,
                    blocks_processed,
                    cache_hits = cache_stats.hits,
                    cache_misses = cache_stats.misses,
                    hit_rate_pct = format!("{:.2}", cache_stats.hit_rate_percent()),
                    "Batch ingestion failed — cache stats at time of failure"
                );
                return Err(e).with_context(|| {
                    format!("Failed to ingest batch {}-{}", current_height, batch_end)
                });
            }
        };

        shutdown_tracker.record_batch_commit(batch_end);

        // Check for shutdown after batch completes (batch can take a long time)
        if shutdown_token.is_cancelled() {
            tracing::info!(
                processed_up_to = batch_end,
                "Shutdown requested after batch, stopping catchup"
            );
            break;
        }

        blocks_processed += blocks.len() as u64;
        current_height = batch_end + 1;

        // Log progress with ETA and entity counts
        let cache_stats = orchestrator.cache_stats();
        let total_elapsed = start_time.elapsed().as_secs_f64();
        let overall_bps = blocks_processed as f64 / total_elapsed;
        let behind = tip.saturating_sub(batch_end);
        let eta_secs = if overall_bps > 0.0 {
            behind as f64 / overall_bps
        } else {
            0.0
        };
        let progress_pct = if tip > 0 {
            (batch_end as f64 / tip as f64) * 100.0
        } else {
            0.0
        };

        tracing::info!(
            height = batch_end,
            blocks = blocks_processed,
            txs = ingestion_stats.transactions,
            avg_bps = format!("{:.1}", overall_bps),
            batch_secs = format!("{:.1}", ingest_elapsed.as_secs_f64()),
            cache_hit_rate = format!("{:.1}%", cache_stats.hit_rate_percent()),
            progress = format!("{:.1}%", progress_pct),
            behind,
            eta = format_eta(eta_secs),
            "Batch ingested"
        );

        // Periodic cache stats logging (every 10k ops, avoids log flooding)
        orchestrator.maybe_log_cache_stats();
    }

    // ═══════════════════════════════════════════════════════════════
    // PHASE B: TRANSITION
    // ═══════════════════════════════════════════════════════════════
    let catchup_elapsed = start_time.elapsed();
    tracing::info!(
        blocks = blocks_processed,
        duration_secs = format!("{:.1}", catchup_elapsed.as_secs_f64()),
        "Catchup phase complete"
    );

    if max_height != u32::MAX && current_height > max_height {
        println!("\n✅ Reached max_height {}. Exiting.", max_height);
        return Ok(());
    }

    // ═══════════════════════════════════════════════════════════════
    // PHASE C: REAL-TIME - Listen for new blocks via ZMQ
    // ═══════════════════════════════════════════════════════════════
    println!("\n👂 Phase C: Real-time mode via ZMQ...");
    println!("   Listening at: {}", rpc_config.zmq_endpoint);
    println!("   Press Ctrl+C to stop\n");

    // Establish persistent ZMQ connection (stays open across blocks)
    let mut zmq_listener = ZmqBlockListener::from_config(rpc_config);
    zmq_listener
        .connect()
        .await
        .context("Failed to establish ZMQ connection")?;

    let max_consecutive_failures = rpc_config.zmq_max_consecutive_failures;
    let mut consecutive_failures: u32 = 0;

    loop {
        tracing::info!("Waiting for new block via ZMQ...");

        // Wait for ZMQ notification with graceful shutdown support
        let block_hash = tokio::select! {
            _ = shutdown_token.cancelled() => {
                tracing::info!("Shutdown signal received in ZMQ listener");
                break;
            }
            result = zmq_listener.recv_block_hash() => {
                match result {
                    Ok(hash) => {
                        consecutive_failures = 0;
                        hash
                    }
                    Err(e) => {
                        consecutive_failures += 1;
                        tracing::error!(
                            error = %e,
                            consecutive_failures,
                            max = max_consecutive_failures,
                            "ZMQ recv failed"
                        );

                        if consecutive_failures >= max_consecutive_failures {
                            return Err(e.context(format!(
                                "ZMQ listener failed {} consecutive times, giving up",
                                consecutive_failures
                            )));
                        }

                        // Cancellable backoff: allow shutdown to interrupt retry sleep.
                        tokio::select! {
                            _ = shutdown_token.cancelled() => {
                                tracing::info!("Shutdown signal received during ZMQ retry backoff");
                                break;
                            }
                            _ = tokio::time::sleep(Duration::from_secs(2)) => {}
                        }
                        continue;
                    }
                }
            }
        };

        tracing::info!(
            block_hash = %block_hash,
            "New block notification received"
        );

        // Fetch the new tip height and ingest any new blocks
        // (handles multi-block gaps if we somehow missed one)
        let tip = provider
            .get_tip_height()
            .await
            .context("Failed to get chain tip after ZMQ notification")?;

        // Enforce max_height in real-time mode
        let effective_tip = tip.min(max_height);

        if current_height <= effective_tip {
            let new_blocks = effective_tip - current_height + 1;
            tracing::info!(
                new_blocks,
                from = current_height,
                to = effective_tip,
                "Ingesting new blocks"
            );

            let mut height = current_height;
            while height <= effective_tip {
                let block_data = provider
                    .get_block(height)
                    .await
                    .with_context(|| format!("Failed to fetch block {} via RPC", height))?;

                if let Some(block_tuple) = block_data {
                    match orchestrator.ingest_blocks_batch(&[block_tuple]).await {
                        Ok(_batch_stats) => {
                            shutdown_tracker.record_block_commit(height);
                            blocks_processed += 1;
                            tracing::info!(
                                height,
                                total_blocks = blocks_processed,
                                "Block ingested (real-time)"
                            );
                            // Periodic cache stats logging
                            orchestrator.maybe_log_cache_stats();
                            height += 1;
                        }
                        Err(WriterError::ReorgDetected {
                            height: reorg_height,
                            expected,
                            actual,
                        }) => {
                            tracing::warn!(
                                reorg_height,
                                expected = %expected,
                                actual = %actual,
                                "Chain reorganization detected — initiating rollback"
                            );

                            // Find fork point by walking back
                            let fork_point = find_fork_point(
                                &orchestrator,
                                &provider,
                                reorg_height.saturating_sub(1),
                            )
                            .await
                            .context("Failed to find fork point during reorg")?;

                            tracing::warn!(
                                fork_point,
                                blocks_to_rollback = reorg_height.saturating_sub(1) - fork_point,
                                "Fork point found, rolling back"
                            );

                            // Roll back from current tip to fork point
                            let rolled_back = orchestrator
                                .rollback_to_height(reorg_height.saturating_sub(1), fork_point)
                                .await
                                .context("Rollback failed during reorg handling")?;

                            tracing::warn!(
                                rolled_back,
                                new_tip = fork_point,
                                "Rollback complete, re-ingesting canonical chain"
                            );

                            // Resume from fork point + 1
                            // The outer ZMQ loop will re-fetch the tip on next iteration
                            height = fork_point + 1;
                            // Update effective_tip for the while loop bound
                            // (we break and let the outer loop re-fetch)
                            current_height = height;
                            break;
                        }
                        Err(e) => {
                            return Err(e)
                                .with_context(|| format!("Failed to ingest block {}", height));
                        }
                    }
                } else {
                    height += 1;
                }
            }

            if height > effective_tip {
                // Normal completion (no reorg)
                current_height = effective_tip + 1;
            }
        }

        // Periodic ZMQ health stats
        if blocks_processed % 10 == 0 && blocks_processed > 0 {
            let zmq_stats = zmq_listener.stats();
            tracing::info!(
                zmq_messages = zmq_stats.messages_received,
                zmq_reconnections = zmq_stats.reconnections,
                zmq_errors = zmq_stats.recv_errors,
                zmq_sequence_gaps = zmq_stats.sequence_gaps,
                total_blocks = blocks_processed,
                "ZMQ listener health"
            );
        }

        // Exit if we've reached the user-specified max_height
        if max_height != u32::MAX && current_height > max_height {
            println!(
                "\n✅ Reached max_height {} in real-time mode. Exiting.",
                max_height
            );
            break;
        }
    }

    // Graceful shutdown: save cache to disk using tracked commit height
    if !cache_file.is_empty() {
        if let Some(save_height) = shutdown_tracker.save_height() {
            match orchestrator
                .get_cache()
                .save_to_file(cache_file, save_height)
            {
                Ok(saved) => {
                    tracing::info!(
                        entries = saved,
                        height = save_height,
                        "UTXO cache saved on shutdown"
                    );
                }
                Err(e) => {
                    tracing::error!(error = %e, "Failed to save UTXO cache on shutdown");
                }
            }
        } else {
            tracing::info!("No blocks committed this session — preserving existing cache file");
        }
    }

    // Graceful shutdown: log final stats
    let zmq_stats = zmq_listener.stats();
    let total_elapsed = start_time.elapsed();
    tracing::info!(
        total_blocks = blocks_processed,
        last_height = current_height.saturating_sub(1),
        duration_secs = format!("{:.1}", total_elapsed.as_secs_f64()),
        zmq_messages = zmq_stats.messages_received,
        zmq_reconnections = zmq_stats.reconnections,
        zmq_errors = zmq_stats.recv_errors,
        zmq_sequence_gaps = zmq_stats.sequence_gaps,
        "Live ingestion shutdown complete"
    );

    Ok(())
}

/// Find the fork point between our stored chain and Bitcoin Core's canonical chain
///
/// Walks backwards from `start_height`, comparing our stored block hashes
/// with the canonical hashes from Bitcoin Core via RPC. Returns the height
/// of the last block where both chains agree.
///
/// Format an ETA in seconds as a human-readable string (e.g. "25h24m", "42m").
fn format_eta(secs: f64) -> String {
    if secs <= 0.0 {
        return "n/a".to_string();
    }
    let h = (secs / 3600.0) as u64;
    let m = ((secs % 3600.0) / 60.0) as u64;
    if h > 0 {
        format!("{}h{}m", h, m)
    } else {
        format!("{}m", m)
    }
}

/// Used during chain reorganization handling to determine how far back
/// to roll back before re-ingesting the canonical chain.
async fn find_fork_point<W: GraphWriter + 'static>(
    orchestrator: &IngestionOrchestrator<W>,
    provider: &RpcBlockProvider,
    start_height: u32,
) -> Result<u32> {
    for height in (0..=start_height).rev() {
        let our_hash = orchestrator
            .lookup_block_hash(height)
            .await
            .with_context(|| format!("Failed to lookup stored block hash at height {}", height))?;

        let canonical_hash = provider
            .get_block_hash(height)
            .await
            .with_context(|| format!("Failed to get canonical block hash at height {}", height))?;

        if our_hash.as_deref() == Some(&canonical_hash) {
            tracing::info!(
                height,
                hash = %canonical_hash,
                "Fork point found — chains agree at this height"
            );
            return Ok(height);
        }

        tracing::debug!(
            height,
            our_hash = ?our_hash,
            canonical_hash = %canonical_hash,
            "Chain mismatch, checking lower height"
        );
    }

    // If we get here, even genesis blocks don't match. This means either:
    // - The database was built against a different network (e.g., testnet vs mainnet)
    // - The database is corrupted
    // Returning Ok(0) would be wrong — it would roll back to genesis and try to
    // re-ingest, but since genesis doesn't match, it would fail again in a loop.
    anyhow::bail!(
        "Fork point not found — stored and canonical chains diverge all the way to genesis. \
         This likely means the database was built against a different Bitcoin network."
    );
}
